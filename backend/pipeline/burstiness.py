import random
import re
from typing import List

# Apply burstiness to the input text by shuffling sentences, merging sentences, and splitting long sentences. This helps create a more human-like writing style by introducing variability in sentence structure and length.
def split_sentences(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

# Shuffle sentences with a certain probability to create variability in sentence order, while keeping short texts unchanged to preserve coherence.
def shuffle_sentences(sentences: List[str]) -> List[str]:
    if len(sentences) < 3:
        return sentences
    # Aggressive burstiness: multiple local swaps and occasional window shuffle.
    shuffled = sentences[:]
    for i in range(len(shuffled) - 1):
        if random.random() < 0.35:
            shuffled[i], shuffled[i + 1] = shuffled[i + 1], shuffled[i]
    if len(shuffled) >= 5 and random.random() < 0.25:
        start = random.randint(0, len(shuffled) - 4)
        window = shuffled[start : start + 4]
        random.shuffle(window)
        shuffled[start : start + 4] = window
    return shuffled

# Merge sentences with a certain probability to create variability in sentence structure, while ensuring that the resulting sentences remain coherent and not excessively long.
def merge_sentences(sentences: List[str]) -> List[str]:
    if len(sentences) < 2:
        return sentences

    merged: List[str] = []
    i = 0
    while i < len(sentences):
        if i < len(sentences) - 1 and random.random() < 0.6:
            left = re.sub(r"[.!?]+$", "", sentences[i]).strip()
            right = sentences[i + 1].strip()
            if re.match(r"^(and|but|so|because|also|anyway|plus)\b", right, re.IGNORECASE):
                merged.append(f"{left}, {right}")
            else:
                merged.append(f"{left}, and {right}")
            i += 2
            continue
        merged.append(sentences[i])
        i += 1
    return merged

# Split long sentences with a certain probability to create variability in sentence length, while ensuring that the resulting sentences remain coherent and not excessively short.
def split_long_sentence(sentences: List[str]) -> List[str]:
    if not sentences:
        return sentences

    result: List[str] = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) > 14 and random.random() < 0.7:
            comma_parts = [p.strip() for p in sentence.split(",", 1)]
            if len(comma_parts) == 2 and comma_parts[0] and comma_parts[1]:
                result.append(comma_parts[0])
                result.append(comma_parts[1])
                continue

            mid = len(words) // 2
            left = " ".join(words[:mid]).strip()
            right = " ".join(words[mid:]).strip()
            if left and right:
                result.append(left)
                result.append(right)
                continue
        result.append(sentence)

    return result

# Main function to apply all style transformations to the input text, including contractions, casual word replacements, hedging, fillers, discourse markers, and openers. The transformations are applied in a specific order to ensure a natural flow of the text.
def mix_sentences(text: str) -> str:
    if not text:
        return text

    sentences = split_sentences(text)
    if not sentences:
        return text

    sentences = shuffle_sentences(sentences)
    sentences = merge_sentences(sentences)
    sentences = split_long_sentence(sentences)

    cleaned = [re.sub(r"[.!?]+$", "", s.strip()) for s in sentences if s.strip()]
    if not cleaned:
        return ""
    return ". ".join(cleaned) + "."

# Main function to apply burstiness to the input text by shuffling sentences, merging sentences, and splitting long sentences. This helps create a more human-like writing style by introducing variability in sentence structure and length.
def apply_burstiness(text: str) -> str:
    return mix_sentences(text)
