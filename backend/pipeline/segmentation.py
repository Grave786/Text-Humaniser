import re
from typing import List

try:
    import nltk
except Exception:  # pragma: no cover
    nltk = None


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> List[str]:
    normalized = clean_text(text)
    if not normalized:
        return []
    return [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]


def _regex_sentence_split(paragraph: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return [s.strip() for s in parts if s.strip()]


def split_sentences(paragraph: str) -> List[str]:
    if not paragraph or not paragraph.strip():
        return []

    if nltk is not None:
        try:
            return [s.strip() for s in nltk.sent_tokenize(paragraph) if s.strip()]
        except Exception:
            pass
    return _regex_sentence_split(paragraph)


def create_chunks(
    sentences: List[str], min_sentences: int = 2, max_sentences: int = 3
) -> List[str]:
    if not sentences:
        return []
    if min_sentences < 1:
        raise ValueError("min_sentences must be >= 1")
    if max_sentences < min_sentences:
        raise ValueError("max_sentences must be >= min_sentences")

    chunk_groups: List[List[str]] = []
    current_sentences: List[str] = []

    for sentence in sentences:
        if not sentence or not sentence.strip():
            continue

        current_sentences.append(sentence.strip())
        if len(current_sentences) == max_sentences:
            chunk_groups.append(current_sentences)
            current_sentences = []

    if current_sentences:
        # Prefer keeping final chunks within the requested range.
        if len(current_sentences) < min_sentences and chunk_groups:
            previous = chunk_groups[-1]
            if len(previous) + len(current_sentences) <= max_sentences:
                previous.extend(current_sentences)
            elif len(previous) > min_sentences:
                current_sentences.insert(0, previous.pop())
                chunk_groups.append(current_sentences)
            else:
                chunk_groups.append(current_sentences)
        else:
            chunk_groups.append(current_sentences)

    chunks = [" ".join(group) for group in chunk_groups]
    return chunks


def segment_text(
    text: str, min_sentences: int = 2, max_sentences: int = 3
) -> List[str]:
    all_chunks: List[str] = []
    for paragraph in split_paragraphs(text):
        sentences = split_sentences(paragraph)
        paragraph_chunks = create_chunks(
            sentences, min_sentences=min_sentences, max_sentences=max_sentences
        )
        all_chunks.extend(paragraph_chunks)
    return all_chunks
