import random
import re
from typing import Dict, List

# Map of words to their variations for lexicon diversity. This is used to randomly replace certain common words with their synonyms or related phrases to create a more human-like writing style by introducing variability in word choice.
VARIATION_MAP: Dict[str, List[str]] = {
    "important": ["key", "notable", "worth noting"],
    "good": ["solid", "useful", "helpful"],
    "bad": ["weak", "poor", "limited"],
    "show": ["demonstrate", "highlight", "point out"],
    "use": ["apply", "use", "work with"],
    "because": ["since", "because", "as"],
    "also": ["also", "plus", "on top of that"],
}

# Optimize perplexity by applying a series of transformations to the input text, including normalizing spacing, collapsing repeated words, varying lexicon, varying sentence rhythm, and polishing end punctuation. The transformations are applied based on a specified intensity level, which controls the likelihood of each type of transformation being applied. The resulting text is normalized for spacing to ensure consistent formatting.
def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

# Normalize spacing by collapsing multiple spaces into one and removing spaces before punctuation. This helps ensure that the resulting text has consistent formatting after transformations are applied.
def _normalize_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()

# Collapse immediate repeated words (e.g., "the the" -> "the") to reduce redundancy and improve readability, which can help optimize perplexity by creating a more natural flow of text.
def _de_repeat(text: str) -> str:
    # Collapse immediate repeated words: "the the" -> "the"
    return re.sub(r"\b(\w+)\s+\1\b", r"\1", text, flags=re.IGNORECASE)

# Vary lexicon by randomly replacing certain common words with their synonyms or related phrases based on the specified intensity level. This helps create a more human-like writing style by introducing variability in word choice.
def _vary_lexicon(text: str, intensity: float) -> str:
    result = text
    for source, choices in VARIATION_MAP.items():
        if random.random() < (0.18 + 0.35 * intensity):
            replacement = random.choice(choices)
            result = re.sub(
                rf"\b{re.escape(source)}\b",
                replacement,
                result,
                count=1,
                flags=re.IGNORECASE,
            )
    return result

# Vary sentence rhythm by shuffling sentences, merging sentences, and splitting long sentences based on the specified intensity level. This helps create a more human-like writing style by introducing variability in sentence structure and length.
def _vary_sentence_rhythm(text: str, intensity: float) -> str:
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return text

    mixed: List[str] = []
    i = 0
    while i < len(sentences):
        # Merge adjacent short sentences to vary cadence.
        if i < len(sentences) - 1 and random.random() < (0.12 + 0.25 * intensity):
            left = re.sub(r"[.!?]+$", "", sentences[i]).strip()
            right = sentences[i + 1].strip()
            mixed.append(f"{left}, {right}")
            i += 2
            continue

        # Split long sentence once around punctuation.
        sentence = sentences[i]
        if len(sentence.split()) > 22 and random.random() < (0.1 + 0.2 * intensity):
            parts = [p.strip() for p in sentence.split(",", 1)]
            if len(parts) == 2 and parts[0] and parts[1]:
                mixed.append(parts[0] + ".")
                mixed.append(parts[1])
            else:
                mixed.append(sentence)
        else:
            mixed.append(sentence)
        i += 1

    return " ".join(mixed).strip()

# Polish end punctuation by ensuring that sentences end with appropriate punctuation marks. If a sentence does not end with punctuation, add a period at the end. This helps improve readability and ensures that the resulting text has proper sentence structure.
def _polish_end_punctuation(text: str) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return text
    polished: List[str] = []
    for s in sentences:
        if re.search(r"[.!?]$", s):
            polished.append(s)
        else:
            polished.append(s + ".")
    return " ".join(polished).strip()

# Main function to optimize perplexity by applying a series of transformations to the input text, including normalizing spacing, collapsing repeated words, varying lexicon, varying sentence rhythm, and polishing end punctuation. The transformations are applied based on a specified intensity level, which controls the likelihood of each type of transformation being applied. The resulting text is normalized for spacing to ensure consistent formatting.
def optimize_perplexity(text: str, intensity: float = 0.35) -> str:
    if not text:
        return text

    intensity = max(0.0, min(1.0, intensity))
    optimized = text
    optimized = _normalize_spacing(optimized)
    optimized = _de_repeat(optimized)
    optimized = _vary_lexicon(optimized, intensity)
    optimized = _vary_sentence_rhythm(optimized, intensity)
    optimized = _normalize_spacing(optimized)
    optimized = _polish_end_punctuation(optimized)
    return optimized
