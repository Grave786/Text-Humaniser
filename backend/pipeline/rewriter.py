import os
from dataclasses import dataclass
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

_BATCH_SIZE = int(os.getenv("REWRITER_BATCH_SIZE", "8") or "8")
_USE_FP16 = os.getenv("REWRITER_FP16", "1").strip().lower() in {"1", "true", "yes", "on"}
_NUM_BEAMS = int(os.getenv("REWRITER_NUM_BEAMS", "1") or "1")
_TEMPERATURE = float(os.getenv("REWRITER_TEMPERATURE", "0.9") or "0.9")
_TOP_P = float(os.getenv("REWRITER_TOP_P", "0.92") or "0.92")
_REPETITION_PENALTY = float(os.getenv("REWRITER_REPETITION_PENALTY", "1.05") or "1.05")
_MAX_INPUT_TOKENS = int(os.getenv("REWRITER_MAX_INPUT_TOKENS", "512") or "512")

# T5-based rewriter class to handle paraphrasing of text segments using a specified T5 model. The class includes methods for loading the model and tokenizer, as well as a method for rewriting text segments while applying safety guards to prevent excessive content collapse.
@dataclass
class RewriteStage:
    model_name: str
    prompt_mode: str  # "raw" | "t5_paraphrase" | "t5_humanize"
    model: Optional[object] = None
    tokenizer: Optional[object] = None
    load_error: Optional[str] = None
    loaded: bool = False

# T5-based rewriter class to handle paraphrasing of text segments using a specified T5 model. The class includes methods for loading the model and tokenizer, as well as a method for rewriting text segments while applying safety guards to prevent excessive content collapse.
class TwoStageRewriter:
    def __init__(self, stage1_model: str, stage2_model: str) -> None:
        self.device = "cpu"
        self.stage1 = RewriteStage(model_name=stage1_model, prompt_mode="raw")
        # "t5-base" is not instruction tuned; a classic task prefix tends to behave better.
        self.stage2 = RewriteStage(model_name=stage2_model, prompt_mode="t5_paraphrase")
# T5-based rewriter class to handle paraphrasing of text segments using a specified T5 model. The class includes methods for loading the model and tokenizer, as well as a method for rewriting text segments while applying safety guards to prevent excessive content collapse.
    def _load_stage(self, stage: RewriteStage) -> None:
        if stage.loaded:
            return
        stage.loaded = True

        if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
            stage.load_error = "transformers import failed"
            return
        if torch is None:
            stage.load_error = "torch import failed"
            return

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            stage.tokenizer = AutoTokenizer.from_pretrained(stage.model_name)
            stage.model = AutoModelForSeq2SeqLM.from_pretrained(stage.model_name)
            stage.model.to(self.device)
            if self.device == "cuda" and _USE_FP16:
                try:
                    stage.model.half()
                except Exception:
                    pass
            stage.model.eval()
        except Exception as exc:
            stage.load_error = str(exc)
            stage.model = None
            stage.tokenizer = None
# T5-based rewriter class to handle paraphrasing of text segments using a specified T5 model. The class includes methods for loading the model and tokenizer, as well as a method for rewriting text segments while applying safety guards to prevent excessive content collapse.
    def _build_prompt(self, stage: RewriteStage, text: str) -> str:
        if stage.prompt_mode == "t5_paraphrase":
            return f"paraphrase: {text}"
        if stage.prompt_mode == "t5_humanize":
            return f"rewrite this text in natural human phrasing: {text}"
        return text

    def _generate_many(self, stage: RewriteStage, texts: list[str], stage_strength: float) -> list[str]:
        self._load_stage(stage)
        if stage.model is None or stage.tokenizer is None:
            return texts

        if torch is None:
            return texts

        prompts = [self._build_prompt(stage, t) for t in texts]
        try:
            tokenized = stage.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=_MAX_INPUT_TOKENS,
            )
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

            # Pick a single max_new_tokens for the batch to keep generate fast.
            # (Per-sample max_new_tokens isn't supported by HF generate.)
            if "attention_mask" in tokenized:
                max_in = int(tokenized["attention_mask"].sum(dim=1).max().item())
            else:
                max_in = int(tokenized["input_ids"].shape[-1])
            max_new_tokens = max(32, min(220, int(max_in * (0.9 + 0.1 * stage_strength))))

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "num_beams": max(1, _NUM_BEAMS),
                "do_sample": True,
                "temperature": max(0.1, _TEMPERATURE + (0.05 * stage_strength)),
                "top_p": min(0.99, max(0.5, _TOP_P)),
                "repetition_penalty": max(1.0, _REPETITION_PENALTY),
                "early_stopping": True,
            }

            # Beams + sampling is usually wasteful; if beams > 1, disable sampling.
            if gen_kwargs["num_beams"] > 1:
                gen_kwargs["do_sample"] = False

            with torch.inference_mode():
                outputs = stage.model.generate(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized.get("attention_mask"),
                    **gen_kwargs,
                )

            decoded = stage.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            rewritten = [(d or "").strip() for d in decoded]
            return [r if r else t for r, t in zip(rewritten, texts)]
        except Exception:
            return texts
# T5-based rewriter class to handle paraphrasing of text segments using a specified T5 model. The class includes methods for loading the model and tokenizer, as well as a method for rewriting text segments while applying safety guards to prevent excessive content collapse.
    def _run_stage(self, stage: RewriteStage, text: str, stage_strength: float) -> str:
        self._load_stage(stage)
        if stage.model is None or stage.tokenizer is None:
            return text

        prompt = self._build_prompt(stage, text)
        try:
            inputs = stage.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)

            input_tokens = int(inputs["input_ids"].shape[-1])
            min_new_tokens = max(24, min(220, int(input_tokens * (0.62 + 0.05 * stage_strength))))
            max_new_tokens = max(min_new_tokens + 16, min(320, int(input_tokens * (1.18 + 0.08 * stage_strength))))

            with torch.no_grad():
                outputs = stage.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    min_new_tokens=min_new_tokens,
                    max_new_tokens=max_new_tokens,
                    num_beams=4,
                    do_sample=True,
                    temperature=0.85 + (0.1 * stage_strength),
                    top_p=0.92,
                    early_stopping=True,
                )

            rewritten = stage.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return rewritten if rewritten else text
        except Exception:
            return text
# T5-based rewriter class to handle paraphrasing of text segments using a specified T5 model. The class includes methods for loading the model and tokenizer, as well as a method for rewriting text segments while applying safety guards to prevent excessive content collapse.
    def rewrite(self, segment: str) -> str:
        if not segment:
            return segment

        original_segment = segment.strip()
        original_words = original_segment.split()
        if not original_words:
            return segment

        # Keep very long single chunks unchanged to avoid truncation artifacts.
        if len(original_words) > 320:
            return original_segment

        stage1_text = self._run_stage(self.stage1, original_segment, stage_strength=1.0)
        stage2_text = self._run_stage(self.stage2, stage1_text, stage_strength=0.6)

        final_text = stage2_text if stage2_text else (stage1_text if stage1_text else original_segment)
        final_count = len(final_text.split())
        if final_count < int(len(original_words) * 0.8):
            return original_segment
        return final_text

    def rewrite_many(self, segments: list[str]) -> list[str]:
        if not segments:
            return []

        # Preserve empty strings and very long segments as-is; batch the rest.
        out: list[str] = [s for s in segments]
        idxs: list[int] = []
        work: list[str] = []
        for i, seg in enumerate(segments):
            if not seg or not seg.strip():
                continue
            if len(seg.split()) > 320:
                continue
            idxs.append(i)
            work.append(seg.strip())

        if not work:
            return out

        # Batch stage 1 and stage 2 to reduce per-call overhead.
        stage1_results: list[str] = []
        for i in range(0, len(work), max(1, _BATCH_SIZE)):
            batch = work[i : i + max(1, _BATCH_SIZE)]
            stage1_results.extend(self._generate_many(self.stage1, batch, stage_strength=1.0))

        stage2_results: list[str] = []
        for i in range(0, len(stage1_results), max(1, _BATCH_SIZE)):
            batch = stage1_results[i : i + max(1, _BATCH_SIZE)]
            stage2_results.extend(self._generate_many(self.stage2, batch, stage_strength=0.6))

        for out_i, original_segment, stage2_text in zip(idxs, work, stage2_results):
            final_text = stage2_text.strip() if stage2_text else original_segment
            if len(final_text.split()) < int(len(original_segment.split()) * 0.8):
                final_text = original_segment
            out[out_i] = final_text
        return out
# Control repetition by removing immediate repeated words and short phrases, ensuring that the resulting text is more concise and avoids unnecessary redundancy while maintaining coherence.
    def status(self) -> dict:
        self._load_stage(self.stage1)
        self._load_stage(self.stage2)
        stage1_loaded = self.stage1.model is not None and self.stage1.tokenizer is not None
        stage2_loaded = self.stage2.model is not None and self.stage2.tokenizer is not None
        return {
            "model_name": f"{self.stage1.model_name} -> {self.stage2.model_name}",
            "loaded": stage1_loaded and stage2_loaded,
            "device": self.device,
            "load_error": self.stage1.load_error or self.stage2.load_error,
            "stage_1": {
                "model_name": self.stage1.model_name,
                "loaded": stage1_loaded,
                "load_error": self.stage1.load_error,
            },
            "stage_2": {
                "model_name": self.stage2.model_name,
                "loaded": stage2_loaded,
                "load_error": self.stage2.load_error,
            },
        }

# Control repetition by removing immediate repeated words and short phrases, ensuring that the resulting text is more concise and avoids unnecessary redundancy while maintaining coherence.
_STAGE1_MODEL_NAME = os.getenv(
    "REWRITER_STAGE1_MODEL",
    os.getenv("T5_MODEL_NAME", "tuner007/pegasus_paraphrase"),
)
_STAGE2_MODEL_NAME = os.getenv("REWRITER_STAGE2_MODEL", "t5-base")

_rewriter = TwoStageRewriter(
    stage1_model=_STAGE1_MODEL_NAME,
    stage2_model=_STAGE2_MODEL_NAME,
)

#  Control repetition by removing immediate repeated words and short phrases, ensuring that the resulting text is more concise and avoids unnecessary redundancy while maintaining coherence.
def warmup_rewriter() -> None:
    _rewriter.status()

# Control repetition by removing immediate repeated words and short phrases, ensuring that the resulting text is more concise and avoids unnecessary redundancy while maintaining coherence.
def get_rewriter_status() -> dict:
    return _rewriter.status()

# Control repetition by removing immediate repeated words and short phrases, ensuring that the resulting text is more concise and avoids unnecessary redundancy while maintaining coherence.
def rewrite_segment(segment: str) -> str:
    return _rewriter.rewrite(segment)


def rewrite_segments(segments: list[str]) -> list[str]:
    return _rewriter.rewrite_many(segments)
