import random
import re
from typing import List

# Style transformation functions to create a more human-like writing style by introducing variability in syntax, contractions, casual word replacements, hedging, fillers, discourse markers, and openers. The main function `syntax_transform_spacy` applies these transformations in a specific order to ensure a natural flow of the text.
def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

# Join sentences with proper spacing and punctuation, ensuring that the resulting text has consistent formatting after style transformations are applied.
def _join_sentences(sentences: List[str]) -> str:
    return " ".join(s.strip() for s in sentences if s.strip()).strip()

# Normalize spacing by collapsing multiple spaces into one and removing spaces before punctuation. This helps ensure that the resulting text has consistent formatting after style transformations are applied.
def _normalize_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()

# Apply paraphrasing using a T5 model to rewrite the input text while preserving its meaning. The function includes safety guards to prevent excessive content collapse and handles exceptions gracefully by returning the original segment if any issues occur during rewriting.
def adjust_pos(text: str) -> str:
    if not text:
        return text

    # Light POS-style balancing: reduce adverb overuse and adjective stacks.
    words = text.split()
    adverb_count = sum(1 for w in words if re.match(r"^[A-Za-z]+ly[.,!?]?$", w))
    if words and (adverb_count / max(len(words), 1)) > 0.1:
        reduced: List[str] = []
        drop_toggle = False
        for w in words:
            if re.match(r"^[A-Za-z]+ly[.,!?]?$", w) and drop_toggle:
                drop_toggle = False
                continue
            if re.match(r"^[A-Za-z]+ly[.,!?]?$", w):
                drop_toggle = True
            reduced.append(w)
        text = " ".join(reduced)

    # Reduce repetitive adjective chains.
    text = re.sub(
        r"\b(very|really|quite)\s+(very|really|quite)\s+",
        r"\1 ",
        text,
        flags=re.IGNORECASE,
    )
    return _normalize_spacing(text)

# T5-based rewriter class to handle paraphrasing of text segments using a specified T5 model. The class includes methods for loading the model and tokenizer, as well as a method for rewriting text segments while applying safety guards to prevent excessive content collapse.
def inject_stopwords(text: str) -> str:
    if not text:
        return text

    sentences = _split_sentences(text)
    if not sentences:
        return text

    injected: List[str] = []
    for sentence in sentences:
        s = sentence
        if len(s.split()) >= 10 and random.random() < 0.35:
            # Insert common stopwords to soften rigid phrasing.
            if re.search(r"\b(I think|I feel|I believe)\s+", s, flags=re.IGNORECASE):
                s = re.sub(r"\b(I think|I feel|I believe)\s+", r"\1 that ", s, count=1, flags=re.IGNORECASE)
            elif "," in s:
                s = s.replace(",", ", and", 1)
            else:
                words = s.split()
                mid = max(2, min(len(words) - 2, len(words) // 2))
                words.insert(mid, random.choice(["and", "that", "the"]))
                s = " ".join(words)
        injected.append(s)

    return _normalize_spacing(_join_sentences(injected))

# Control repetition by removing immediate repeated words and short phrases, ensuring that the resulting text is more concise and avoids unnecessary redundancy while maintaining coherence.
def control_repetition(text: str) -> str:
    if not text:
        return text

    text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text, flags=re.IGNORECASE)

    # Remove immediate repeated short phrases.
    text = re.sub(
        r"\b(\w+\s+\w+\s+\w+)\s+\1\b",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )
    return _normalize_spacing(text)

# T5Rewriter class that loads a T5 model and tokenizer, and provides a method to rewrite text segments using the model. It handles loading errors and device selection, and includes a method to warm up the model during API startup.
def punctuation_engine(text: str) -> str:
    if not text:
        return text

    sentences = _split_sentences(text)
    if not sentences:
        return text

    updated: List[str] = []
    for s in sentences:
        s = s.strip()
        if not re.search(r"[.!?]$", s):
            s = s + "."
        if random.random() < 0.15 and s.endswith("."):
            s = s[:-1] + random.choice([".", "...", "!"])
        updated.append(s)

    return _normalize_spacing(_join_sentences(updated))

# Main function to apply all style transformations to the input text, including contractions, casual word replacements, hedging, fillers, discourse markers, and openers. The transformations are applied in a specific order to ensure a natural flow of the text.
def readability_mixer(text: str) -> str:
    if not text:
        return text

    simplify_map = {
        "utilize": "use",
        "approximately": "about",
        "demonstrate": "show",
        "assistance": "help",
        "purchase": "buy",
    }

    for src, dst in simplify_map.items():
        text = re.sub(rf"\b{src}\b", dst, text, flags=re.IGNORECASE)

    sentences = _split_sentences(text)
    mixed: List[str] = []
    for s in sentences:
        words = s.split()
        if len(words) > 24:
            parts = s.split(",", 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                mixed.append(parts[0].strip() + ".")
                mixed.append(parts[1].strip())
            else:
                mid = len(words) // 2
                mixed.append(" ".join(words[:mid]).strip() + ".")
                mixed.append(" ".join(words[mid:]).strip())
        else:
            mixed.append(s)

    return _normalize_spacing(_join_sentences(mixed))

# Reorder sentences with a certain probability to create variability in sentence structure, while ensuring that the resulting text remains coherent and maintains a logical flow.
def reorder_sentences(text: str) -> str:
    if not text:
        return text
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return text

    reordered = sentences[:]
    for i in range(len(reordered) - 1):
        if random.random() < 0.12:
            reordered[i], reordered[i + 1] = reordered[i + 1], reordered[i]
    return _normalize_spacing(_join_sentences(reordered))
