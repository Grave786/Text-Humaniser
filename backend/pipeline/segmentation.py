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


def create_chunks(sentences: List[str], max_words: int = 45) -> List[str]:
    if not sentences:
        return []
    if max_words < 1:
        raise ValueError("max_words must be >= 1")

    chunks: List[str] = []
    current_words: List[str] = []

    for sentence in sentences:
        sentence_words = sentence.split()
        if not sentence_words:
            continue

        if len(sentence_words) > max_words:
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = []
            for i in range(0, len(sentence_words), max_words):
                chunks.append(" ".join(sentence_words[i : i + max_words]))
            continue

        if len(current_words) + len(sentence_words) <= max_words:
            current_words.extend(sentence_words)
        else:
            chunks.append(" ".join(current_words))
            current_words = sentence_words.copy()

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def segment_text(text: str, max_words: int = 45) -> List[str]:
    all_chunks: List[str] = []
    for paragraph in split_paragraphs(text):
        sentences = split_sentences(paragraph)
        paragraph_chunks = create_chunks(sentences, max_words=max_words)
        all_chunks.extend(paragraph_chunks)
    return all_chunks
