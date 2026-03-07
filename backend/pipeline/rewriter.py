import os
from typing import Optional

try:
    import torch
except Exception:
    torch = None

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except Exception:
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None


class T5Rewriter:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.load_error: Optional[str] = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return

        self._loaded = True

        if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
            self.load_error = "transformers import failed"
            return

        if torch is None:
            self.load_error = "torch import failed"
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

            # Detect device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.model.to(self.device)
            self.model.eval()

        except Exception as exc:
            self.load_error = str(exc)
            self.model = None
            self.tokenizer = None

    def rewrite(self, segment: str) -> str:

        if not segment:
            return segment

        self._load()

        if self.model is None or self.tokenizer is None:
            return segment

        # Limit segment length
        words = segment.split()
        if len(words) > 200:
            segment = " ".join(words[:200])

        prompt = f"paraphrase: {segment}"

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(self.device)

            with torch.no_grad():

                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=128,
                    num_beams=4,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                    early_stopping=True,
                )

            rewritten = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            ).strip()

            return rewritten if rewritten else segment

        except Exception:
            return segment


# Model name from environment variable
_T5_MODEL_NAME = os.getenv("T5_MODEL_NAME", "t5-small")

# Singleton rewriter instance
_rewriter = T5Rewriter(model_name=_T5_MODEL_NAME)


def warmup_rewriter() -> None:
    """
    Load the model during API startup
    to avoid first-request delay.
    """
    _rewriter._load()


def get_rewriter_status() -> dict:
    _rewriter._load()
    return {
        "model_name": _rewriter.model_name,
        "loaded": _rewriter.model is not None and _rewriter.tokenizer is not None,
        "device": _rewriter.device,
        "load_error": _rewriter.load_error,
    }


def rewrite_segment(segment: str) -> str:
    """
    Rewrite a single text segment.
    """
    return _rewriter.rewrite(segment)
