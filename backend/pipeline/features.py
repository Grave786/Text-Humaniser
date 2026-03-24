import os
import re
import string
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from urllib.request import Request, urlopen

import nltk
import numpy as np
import textstat
from nltk.corpus import stopwords
from transformers import AutoTokenizer

MAX_TOKENS = 512
BERT_EMBED_DIM = 768

BERT_BACKEND = os.getenv("BERT_BACKEND", "onnx").strip().lower()
BERT_TOKENIZER_NAME = os.getenv("BERT_TOKENIZER_NAME", "bert-base-uncased").strip()
BERT_ONNX_PATH = os.getenv("BERT_ONNX_PATH", "models/onnx/bert/model.onnx").strip()
BERT_ONNX_URL = os.getenv("BERT_ONNX_URL", "").strip()
BERT_ONNX_DATA_URL = os.getenv("BERT_ONNX_DATA_URL", "").strip()
ONNX_DATA_REQUIRED = os.getenv("ONNX_DATA_REQUIRED", "1").strip().lower() in {"1", "true", "yes", "on"}
ONNX_RUNTIME_DOWNLOAD = os.getenv("ONNX_RUNTIME_DOWNLOAD", "0").strip().lower() in {"1", "true", "yes", "on"}
FREE_TIER_MODE = os.getenv("FREE_TIER_MODE", "1").strip().lower() in {"1", "true", "yes", "on"}
BERT_DISABLED = os.getenv("BERT_DISABLED", "1" if FREE_TIER_MODE else "0").strip().lower() in {"1", "true", "yes", "on"}

ENABLE_PERPLEXITY = os.getenv("ENABLE_PERPLEXITY", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
PERPLEXITY_MODEL_NAME = os.getenv("PERPLEXITY_MODEL_NAME", "gpt2").strip()

# Deployment/perf knobs (safe defaults: no network, CPU-first)
NLTK_DOWNLOAD = os.getenv("NLTK_DOWNLOAD", "0").strip().lower() in {"1", "true", "yes", "on"}
ORT_PROVIDERS = os.getenv("ORT_PROVIDERS", "").strip()
ORT_INTRA_OP_THREADS = int(os.getenv("ORT_INTRA_OP_THREADS", "0") or "0")
ORT_INTER_OP_THREADS = int(os.getenv("ORT_INTER_OP_THREADS", "0") or "0")
ORT_PARALLEL = os.getenv("ORT_PARALLEL", "1").strip().lower() in {"1", "true", "yes", "on"}
ORT_ENABLE_MEM_PATTERN = os.getenv("ORT_ENABLE_MEM_PATTERN", "0").strip().lower() in {"1", "true", "yes", "on"}
ORT_ENABLE_CPU_MEM_ARENA = os.getenv("ORT_ENABLE_CPU_MEM_ARENA", "0").strip().lower() in {"1", "true", "yes", "on"}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Utility function to print warnings to stderr with a consistent prefix for easier debugging and monitoring of the feature extraction process.
def _warn(msg: str) -> None:
    print(f"[feature_extractor] {msg}", file=sys.stderr)

# Utility function to resolve a given path to an absolute path based on the project root. This helps ensure that file paths are correctly handled regardless of the current working directory when the code is executed.
def _resolve_project_path(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (PROJECT_ROOT / pp).resolve()

# Utility function to download a file from a specified URL to a destination path, with error handling and progress reporting. This is used to fetch necessary ONNX model files if they are not already present on the filesystem.
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

# Get BERT embedding for a given text, handling long texts by chunking and averaging embeddings. This function also includes error handling to return a zero vector if embedding fails for any reason, ensuring that the feature extraction process can continue even if the BERT model encounters issues.
@lru_cache(maxsize=1)
def _ensure_onnx_present() -> tuple[Path, Path | None]:
    onnx_path = _resolve_project_path(BERT_ONNX_PATH)
    data_path = onnx_path.with_suffix(onnx_path.suffix + ".data")
    needs_data = ONNX_DATA_REQUIRED or bool(BERT_ONNX_DATA_URL)

    if not onnx_path.exists() or (needs_data and not data_path.exists()):
        if ONNX_RUNTIME_DOWNLOAD and BERT_ONNX_URL:
            _warn(f"ONNX artifacts missing; downloading graph -> {onnx_path}")
            _download(BERT_ONNX_URL, onnx_path)
            if needs_data and BERT_ONNX_DATA_URL:
                _warn(f"ONNX artifacts missing; downloading weights -> {data_path}")
                _download(BERT_ONNX_DATA_URL, data_path)

    must_exist = [onnx_path] + ([data_path] if needs_data else [])
    missing = [str(p) for p in must_exist if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing ONNX artifact(s): "
            + ", ".join(missing)
            + ". Ensure build downloads them via scripts/fetch_onnx.py."
        )

    return onnx_path, (data_path if data_path.exists() else None)

# Utility function to ensure that necessary NLTK resources are available, attempting to download them if they are missing. This is important for the stylometric analysis features that rely on tokenization and stopword lists, and it includes error handling to gracefully degrade functionality if resources cannot be obtained.
def _ensure_nltk_resources() -> bool:
    resources = {
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    }
    ok = True
    for resource_name, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            if not NLTK_DOWNLOAD:
                ok = False
                continue
            try:
                nltk.download(resource_name, quiet=True)
            except Exception:
                ok = False
    return ok

# Clean text by normalizing whitespace and removing excessive newlines and tabs. This helps ensure consistent formatting before further processing.
_NLTK_READY = _ensure_nltk_resources()
try:
    stop_words = set(stopwords.words("english"))
except Exception:
    stop_words = set()
    _NLTK_READY = False

# Detector class that loads a machine learning model from a specified path and provides a method to predict probabilities based on input feature vectors. It includes error handling for model loading and prediction, and ensures that the input feature vector is compatible with the expected feature space of the loaded model.
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?]", "", text)
    return text.strip()

#  Get BERT embedding for a given text, handling long texts by chunking and averaging embeddings. This function also includes error handling to return a zero vector if embedding fails for any reason, ensuring that the feature extraction process can continue even if the BERT model encounters issues.
@lru_cache(maxsize=1)
def _get_bert_tokenizer():
    return AutoTokenizer.from_pretrained(BERT_TOKENIZER_NAME)

#   Get BERT embedding for a given text, handling long texts by chunking and averaging embeddings. This function also includes error handling to return a zero vector if embedding fails for any reason, ensuring that the feature extraction process can continue even if the BERT model encounters issues.
@lru_cache(maxsize=1)
def _get_onnx_session():
    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError("onnxruntime is required for BERT_BACKEND=onnx.") from e

    def _pick_providers() -> list[str]:
        available: list[str]
        try:
            available = list(ort.get_available_providers() or [])
        except Exception:
            available = []

        if ORT_PROVIDERS:
            requested = [p.strip() for p in ORT_PROVIDERS.split(",") if p.strip()]
            picked = [p for p in requested if p in available]
        else:
            preferred_order = [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "DmlExecutionProvider",
                "CoreMLExecutionProvider",
                "OpenVINOExecutionProvider",
                "CPUExecutionProvider",
            ]
            picked = [p for p in preferred_order if p in available]

        if "CPUExecutionProvider" in available and "CPUExecutionProvider" not in picked:
            picked.append("CPUExecutionProvider")
        if not picked:
            picked = ["CPUExecutionProvider"]
        return picked

    onnx_path, _data_path = _ensure_onnx_present()
    sess_options = ort.SessionOptions()
    try:
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    except Exception:
        pass
    try:
        if ORT_INTRA_OP_THREADS > 0:
            sess_options.intra_op_num_threads = ORT_INTRA_OP_THREADS
        if ORT_INTER_OP_THREADS > 0:
            sess_options.inter_op_num_threads = ORT_INTER_OP_THREADS
    except Exception:
        pass
    try:
        if ORT_PARALLEL:
            sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    except Exception:
        pass
    try:
        sess_options.enable_mem_pattern = bool(ORT_ENABLE_MEM_PATTERN)
        sess_options.enable_cpu_mem_arena = bool(ORT_ENABLE_CPU_MEM_ARENA)
    except Exception:
        pass
    return ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=_pick_providers())

# Function to warm up the inference stack by loading the BERT tokenizer and ONNX session (if using ONNX backend) during API startup. This helps reduce latency for the first inference request by ensuring that necessary resources are loaded and ready to use.
def warmup_inference_stack() -> None:
    if BERT_DISABLED:
        return
    _get_bert_tokenizer()
    if BERT_BACKEND == "onnx":
        _get_onnx_session()

# Function to get the status of the rewriter, including model name, load status, device, and any load errors. This can be used for monitoring and debugging purposes to ensure the model is ready for use.
def _onnx_forward(tokenized: dict) -> np.ndarray:
    sess = _get_onnx_session()
    input_names = {i.name for i in sess.get_inputs()}
    ort_inputs = {}
    for k, v in tokenized.items():
        if k in input_names:
            ort_inputs[k] = np.asarray(v, dtype=np.int64)
    outputs = sess.run(None, ort_inputs)
    out_meta = sess.get_outputs()
    for _meta, arr in zip(out_meta, outputs):
        a = np.asarray(arr)
        if a.ndim == 3:
            return a
    return np.asarray(outputs[0])

# Get BERT embedding for a given text, handling long texts by chunking and averaging embeddings. This function also includes error handling to return a zero vector if embedding fails for any reason, ensuring that the feature extraction process can continue even if the BERT model encounters issues.
def _hf_forward(tokenized: dict) -> np.ndarray:
    import torch
    from transformers import AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(BERT_TOKENIZER_NAME).to(device).eval()
    inputs = {k: v.to(device) for k, v in tokenized.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state.detach().cpu().numpy()

# Get BERT embedding for a given text, handling long texts by chunking and averaging embeddings. This function also includes error handling to return a zero vector if embedding fails for any reason, ensuring that the feature extraction process can continue even if the BERT model encounters issues.
def get_bert_embedding(text: str) -> np.ndarray:
    if BERT_DISABLED:
        return np.zeros(BERT_EMBED_DIM, dtype=np.float32)

    text = clean_text(text)
    if not text:
        return np.zeros(BERT_EMBED_DIM, dtype=np.float32)

    embeddings = []
    tokenizer = _get_bert_tokenizer()

    for i in range(0, len(text), 2000):
        chunk_text = text[i : i + 2000]
        tokenized = tokenizer(
            chunk_text,
            return_tensors="np",
            truncation=True,
            padding=True,
            max_length=MAX_TOKENS,
        )
        try:
            last_hidden = _hf_forward(tokenized) if BERT_BACKEND == "hf" else _onnx_forward(tokenized)
        except Exception as e:
            _warn(f"BERT embedding failed ({BERT_BACKEND}); returning zeros. Error: {e!r}")
            return np.zeros(BERT_EMBED_DIM, dtype=np.float32)

        cls_embedding = np.asarray(last_hidden[:, 0, :], dtype=np.float32).squeeze()
        if cls_embedding.ndim != 1:
            cls_embedding = np.ravel(cls_embedding)

        if cls_embedding.shape[0] < BERT_EMBED_DIM:
            padded = np.zeros(BERT_EMBED_DIM, dtype=np.float32)
            padded[: cls_embedding.shape[0]] = cls_embedding
            cls_embedding = padded
        elif cls_embedding.shape[0] > BERT_EMBED_DIM:
            cls_embedding = cls_embedding[:BERT_EMBED_DIM]

        embeddings.append(cls_embedding)

    return np.mean(np.vstack(embeddings), axis=0).astype(np.float32)


@lru_cache(maxsize=1)
def _get_gpt2_resources():
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = GPT2Tokenizer.from_pretrained(PERPLEXITY_MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(PERPLEXITY_MODEL_NAME).to(device)
    model.eval()
    return tok, model, device, torch

# Compute perplexity of a given text using a GPT-2 model, with error handling to return a default value if the model cannot be loaded or if an error occurs during computation. This provides an additional feature for stylometric analysis that can help differentiate between human and AI-generated text based on how predictable the text is according to the language model.
def compute_perplexity(text: str) -> float:
    if not ENABLE_PERPLEXITY:
        return 0.0

    tok, model, device, torch = _get_gpt2_resources()
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        return float(torch.exp(outputs.loss).item())

# Clean text by normalizing whitespace and removing excessive newlines and tabs. This helps ensure consistent formatting before further processing.
def _sent_tokenize(text: str) -> list[str]:
    if _NLTK_READY:
        try:
            return nltk.sent_tokenize(text)
        except Exception:
            pass
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    return sents

# Clean text by normalizing whitespace and removing excessive newlines and tabs. This helps ensure consistent formatting before further processing.
def _word_tokenize(text: str) -> list[str]:
    if _NLTK_READY:
        try:
            return nltk.word_tokenize(text)
        except Exception:
            pass
    return re.findall(r"[A-Za-z]+", text)

# Stylometric analysis function that extracts various features from the input text, such as word count, sentence count, average sentence length, stopword ratio, unigram and bigram repetition, part-of-speech ratios, punctuation ratio, and readability score. This function is designed to provide a comprehensive set of features for analyzing the writing style of the text, which can be used in machine learning models to help differentiate between human and AI-generated content.
def stylometric_analysis_no_perplexity(text: str) -> np.ndarray:
    sentences = _sent_tokenize(text)
    words = [w.lower() for w in _word_tokenize(text) if w.isalpha()]

    num_words = len(words)
    num_sentences = len(sentences)

    sent_lengths = [len(_word_tokenize(s)) for s in sentences if s.strip()]
    avg_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 0.0
    sent_len_var = float(np.var(sent_lengths)) if len(sent_lengths) > 1 else 0.0
    burstiness = sent_len_var / avg_sent_len if avg_sent_len > 0 else 0.0

    stopword_ratio = (sum(1 for w in words if w in stop_words) / num_words) if num_words > 0 else 0.0
    unigram_rep = 1 - (len(set(words)) / num_words) if num_words > 0 else 0.0

    bigrams = list(nltk.bigrams(words))
    bigram_counts = Counter(bigrams)
    bigram_rep = (sum(1 for c in bigram_counts.values() if c > 1) / len(bigram_counts)) if bigram_counts else 0.0

    if _NLTK_READY:
        try:
            pos_tags = nltk.pos_tag(words)
            pos_counts = Counter(tag for _, tag in pos_tags)
        except Exception:
            pos_counts = Counter()
    else:
        pos_counts = Counter()

    noun_ratio = sum(pos_counts[t] for t in ["NN", "NNS", "NNP", "NNPS"]) / num_words if num_words else 0.0
    verb_ratio = sum(pos_counts[t] for t in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]) / num_words if num_words else 0.0
    adj_ratio = sum(pos_counts[t] for t in ["JJ", "JJR", "JJS"]) / num_words if num_words else 0.0

    punct_ratio = sum(1 for c in text if c in string.punctuation) / max(len(text), 1)
    readability = float(textstat.flesch_reading_ease(text))

    return np.array(
        [
            num_words,
            num_sentences,
            avg_sent_len,
            sent_len_var,
            burstiness,
            stopword_ratio,
            unigram_rep,
            bigram_rep,
            noun_ratio,
            verb_ratio,
            adj_ratio,
            punct_ratio,
            readability,
        ],
        dtype=np.float32,
    )

# Stylometric analysis function that extracts various features from the input text, such as word count, sentence count, average sentence length, stopword ratio, unigram and bigram repetition, part-of-speech ratios, punctuation ratio, readability score, and perplexity. This function is designed to provide a comprehensive set of features for analyzing the writing style of the text, which can be used in machine learning models to help differentiate between human and AI-generated content.
def stylometric_analysis(text: str) -> np.ndarray:
    base = stylometric_analysis_no_perplexity(text)
    perplexity = compute_perplexity(text)
    return np.concatenate([base, np.array([perplexity], dtype=np.float32)])


@lru_cache(maxsize=512)
def _get_chunk_heavy_features_cached(chunk_text: str) -> tuple[np.ndarray, float]:
    bert_features = get_bert_embedding(chunk_text).astype(np.float32)
    perplexity = compute_perplexity(chunk_text)
    return bert_features, perplexity

# Function to build a feature vector for a sentence that includes both stylometric features of the sentence itself and contextual features from the chunk it belongs to. This combines the BERT embedding and perplexity of the chunk with the stylometric analysis of the individual sentence, providing a richer feature representation for machine learning models that aim to classify or analyze text at the sentence level while considering its context.
def build_features_with_chunk_context(sentence_text: str, chunk_text: str) -> np.ndarray:
    chunk_bert, chunk_perplexity = _get_chunk_heavy_features_cached(chunk_text)
    sentence_style = stylometric_analysis_no_perplexity(sentence_text).astype(np.float32)
    style_with_perplexity = np.concatenate([sentence_style, np.array([chunk_perplexity], dtype=np.float32)])
    return np.concatenate([chunk_bert, style_with_perplexity]).astype(np.float32)

# Function to build a feature vector for a sentence based solely on its own stylometric features, without considering the context of the chunk it belongs to. This provides a more lightweight feature representation that can be used in scenarios where computational resources are limited or when contextual features are not deemed necessary for the analysis.
@lru_cache(maxsize=2048)
def _extract_features_cached(text: str) -> np.ndarray:
    bert_features = get_bert_embedding(text)
    style_features = stylometric_analysis(text)
    return np.concatenate([bert_features, style_features]).astype(np.float32)

# Function to extract features from a given text, combining BERT embeddings and stylometric analysis. This function is designed to be the main entry point for feature extraction, providing a comprehensive feature vector that can be used in machine learning models for tasks such as classification or analysis of text. It includes caching to improve performance on repeated inputs and ensures that the feature extraction process is efficient and robust.
def extract_features(text: str) -> np.ndarray:
    return _extract_features_cached(text).copy()
