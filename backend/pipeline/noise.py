import random
import re
from typing import Dict, List

# Noise injection functions to create a more human-like writing style by introducing variability in sentence structure, punctuation, contractions, and typos. The main function `inject_human_noise` applies these transformations based on a specified intensity level, which controls the likelihood of each type of noise being applied.
CONTRACTION_MAP: Dict[str, str] = {
    "do not": "don't",
    "does not": "doesn't",
    "did not": "didn't",
    "cannot": "can't",
    "can not": "can't",
    "it is": "it's",
    "that is": "that's",
    "there is": "there's",
    "I am": "I'm",
    "I have": "I've",
    "I will": "I'll",
}

# Split text into sentences using regex, ensuring that sentences are properly separated based on punctuation and whitespace. This function is used internally for applying noise transformations at the sentence level.
def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

# Normalize spacing by collapsing multiple spaces into one and removing spaces before punctuation. This helps ensure that the resulting text has consistent formatting after noise transformations are applied.
def _normalize_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()

# Apply contraction noise by randomly replacing common phrases with their contracted forms based on the specified intensity level. This helps create a more casual and human-like writing style.
def _apply_contraction_noise(text: str, intensity: float) -> str:
    noisy = text
    for src, dst in CONTRACTION_MAP.items():
        if random.random() < (0.2 + 0.4 * intensity):
            noisy = re.sub(rf"\b{re.escape(src)}\b", dst, noisy, flags=re.IGNORECASE)
    return noisy

# Apply punctuation noise by randomly altering sentence-ending punctuation based on the specified intensity level. This helps create variability in sentence structure and adds a more human-like touch to the text.
def _apply_punctuation_noise(text: str, intensity: float) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return text

    for i, sentence in enumerate(sentences):
        if random.random() < (0.08 + 0.18 * intensity):
            if sentence.endswith("."):
                sentences[i] = sentence[:-1] + random.choice(["...", "!", "."])
            elif sentence.endswith("!"):
                sentences[i] = sentence[:-1] + random.choice(["!", "..."])
    return " ".join(sentences).strip()

# Apply typo noise by randomly swapping adjacent letters in words that are at least 5 characters long, based on the specified intensity level. This helps create a more human-like writing style by introducing occasional typos that are common in natural writing.
def _apply_typo_noise(text: str, intensity: float) -> str:
    words = text.split()
    if not words:
        return text
# Identify candidate words for typos (at least 5 characters long) and randomly apply a letter swap to create a typo, with a certain probability based on the intensity level. The function also includes a human-like correction pattern where the typo is presented alongside the correct word (e.g., "teh -> the") to mimic common writing mistakes and corrections.
    candidates = [i for i, w in enumerate(words) if len(re.sub(r"[^A-Za-z]", "", w)) >= 5]
    if not candidates or random.random() >= (0.06 + 0.16 * intensity):
        return text
# Randomly select a candidate word and apply a letter swap to create a typo, ensuring that the resulting word still has at least 2 letters to swap. The function also includes a human-like correction pattern where the typo is presented alongside the correct word (e.g., "teh -> the") to mimic common writing mistakes and corrections.
    idx = random.choice(candidates)
    word = words[idx]
    letters = list(word)
    alpha_positions = [j for j, ch in enumerate(letters) if ch.isalpha()]
    if len(alpha_positions) < 2:
        return text
# Randomly select two adjacent letters to swap, ensuring that the resulting word still has at least 2 letters to swap. The function also includes a human-like correction pattern where the typo is presented alongside the correct word (e.g., "teh -> the") to mimic common writing mistakes and corrections.
    p = random.choice(alpha_positions[:-1])
    q = alpha_positions[alpha_positions.index(p) + 1]
    letters[p], letters[q] = letters[q], letters[p]
    typo = "".join(letters)

    # Keep typos subtle: never include explicit correction markers like "typo -> word".
    words[idx] = typo
    return " ".join(words)

# Apply hesitation noise by randomly inserting hesitation phrases (e.g., "uh,", "hmm,") before sentences based on the specified intensity level. This helps create a more human-like writing style by introducing natural hesitations that occur in spoken language.
def _apply_hesitation_noise(text: str, intensity: float) -> str:
    sentences = _split_sentences(text)
    if len(sentences) < 2 or random.random() >= (0.1 + 0.25 * intensity):
        return text
# Randomly select a sentence (not the first one) and insert a hesitation phrase before it, ensuring that the selected sentence does not already start with a hesitation phrase. This helps create a more human-like writing style by introducing natural hesitations that occur in spoken language.
    idx = random.randint(1, len(sentences) - 1)
    hesitations = ("uh,", "hmm,", "well,", "okay,")
    if not re.match(r"^(uh|hmm|well|okay)\b", sentences[idx], re.IGNORECASE):
        sentences[idx] = f"{random.choice(hesitations)} {sentences[idx]}"
    return " ".join(sentences).strip()

# Main function to apply all noise transformations to the input text, including contractions, punctuation, typos, and hesitations. The transformations are applied based on a specified intensity level, which controls the likelihood of each type of noise being applied. The resulting text is normalized for spacing to ensure consistent formatting.
def inject_human_noise(text: str, intensity: float = 0.35, allow_typos: bool = False) -> str:
    if not text:
        return text

    intensity = max(0.0, min(1.0, intensity))
    noisy = text
    noisy = _apply_contraction_noise(noisy, intensity)
    noisy = _apply_punctuation_noise(noisy, intensity)
    if allow_typos:
        noisy = _apply_typo_noise(noisy, intensity)
    noisy = _apply_hesitation_noise(noisy, intensity)
    return _normalize_spacing(noisy)
