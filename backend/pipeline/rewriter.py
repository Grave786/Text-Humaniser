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
        self.stage2 = RewriteStage(model_name=stage2_model, prompt_mode="t5_humanize")
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
