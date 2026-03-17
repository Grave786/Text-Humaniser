import random
import re
from typing import Dict, List, Tuple
from functools import lru_cache

try:
    import spacy
except Exception:
    spacy = None

# This module provides functions to transform text into a more casual style by applying contractions, replacing formal words with casual alternatives, adding hedging phrases, inserting filler words, adding discourse markers, and merging sentences. The transformations are applied with certain probabilities to create a more natural and varied output.
CONTRACTIONS: Dict[str, str] = {
    "do not": "don't",
    "does not": "doesn't",
    "did not": "didn't",
    "cannot": "can't",
    "can not": "can't",
    "will not": "won't",
    "would not": "wouldn't",
    "should not": "shouldn't",
    "could not": "couldn't",
    "is not": "isn't",
    "are not": "aren't",
    "was not": "wasn't",
    "were not": "weren't",
    "has not": "hasn't",
    "have not": "haven't",
    "had not": "hadn't",
    "it is": "it's",
    "that is": "that's",
    "there is": "there's",
    "I am": "I'm",
    "I have": "I've",
    "I will": "I'll",
}

# A mapping of formal words to their casual alternatives, which are applied to the text to make it sound more conversational and less formal.
CASUAL_MAP: Dict[str, str] = {
    "therefore": "so",
    "however": "but",
    "moreover": "also",
    "furthermore": "also",
    "utilize": "use",
    "approximately": "about",
    "demonstrate": "show",
    "assistance": "help",
    "purchase": "buy",
    "regarding": "about",
}
# Lists of hedging phrases, filler words, discourse markers, and openers that can be randomly inserted into the text to create a more casual and human-like style.
HEDGES: Tuple[str, ...] = ("I think", "maybe", "in my view", "it seems")
# Filler words and phrases that can be randomly added to sentences to make the text sound more natural and less formal.
FILLERS: Tuple[str, ...] = ("you know", "to be honest", "kind of", "basically")
# Discourse markers that can be randomly inserted before sentences to create a more conversational flow and add emphasis or transition between ideas.
DISCOURSE_MARKERS: Tuple[str, ...] = ("Also", "Anyway", "Plus", "On top of that")
# Openers that can be randomly added at the beginning of the text to create a more engaging and casual introduction to the content.
OPENERS: Tuple[str, ...] = ("Honestly,", "To be fair,", "In practice,", "At the end of the day,")
CONNECTORS: Tuple[str, ...] = ("and", "but", "so", "because")
IRREGULAR_PARTICIPLES: Dict[str, str] = {
    "written": "wrote",
    "built": "built",
    "made": "made",
    "bought": "bought",
    "found": "found",
    "sent": "sent",
}
LOWERABLE_START_WORDS = {
    "it",
    "this",
    "that",
    "these",
    "those",
    "they",
    "we",
    "you",
    "he",
    "she",
    "there",
}


@lru_cache(maxsize=1)
def _get_spacy_nlp():
    if spacy is None:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return None

# Main function to apply all style transformations to the input text, including contractions, casual word replacements, hedging, fillers, discourse markers, and openers. The transformations are applied in a specific order to ensure a natural flow of the text.
def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

# The main function to apply all style transformations to the input text, including contractions, casual word replacements, hedging, fillers, discourse markers, and openers. The transformations are applied in a specific order to ensure a natural flow of the text.
def _join_sentences(sentences: List[str]) -> str:
    return " ".join(s.strip() for s in sentences if s.strip()).strip()

# Main function to apply all style transformations to the input text, including contractions, casual word replacements, hedging, fillers, discourse markers, and openers. The transformations are applied in a specific order to ensure a natural flow of the text.
def _lower_start_if_safe(sentence: str) -> str:
    if not sentence:
        return sentence
    match = re.match(r"^([A-Za-z']+)", sentence.strip())
    if not match:
        return sentence
    first = match.group(1)
    if first.lower() in LOWERABLE_START_WORDS and len(sentence) > 1:
        return sentence[0].lower() + sentence[1:]
    return sentence

# Main function to apply all style transformations to the input text, including contractions, casual word replacements, hedging, fillers, discourse markers, and openers. The transformations are applied in a specific order to ensure a natural flow of the text.
def apply_contractions(text: str) -> str:
    for src, dst in CONTRACTIONS.items():
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text, flags=re.IGNORECASE)
    return text

# Main function to apply all style transformations to the input text, including contractions, casual word replacements, hedging, fillers, discourse markers, and openers. The transformations are applied in a specific order to ensure a natural flow of the text.
def casualize(text: str) -> str:
    for src, dst in CASUAL_MAP.items():
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text, flags=re.IGNORECASE)
    return text

# Main function to apply all style transformations to the input text, including contractions, casual word replacements, hedging, fillers, discourse markers, and openers. The transformations are applied in a specific order to ensure a natural flow of the text.
def transform_voice(text: str) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return text
# Apply passive to active and active to passive transformations with certain probabilities to create variability in sentence structure while ensuring that the resulting sentences remain coherent and grammatically correct.
    transformed: List[str] = []
    for sentence in sentences:
        s = sentence.strip()

        # Passive -> Active: "The report was written by Alice." -> "Alice wrote the report."
        passive_match = re.match(
            r"^(The|A|An)\s+(.+?)\s+was\s+([A-Za-z]+)\s+by\s+([A-Z][A-Za-z'-]+)([.!?]?)$",
            s,
        )
        if passive_match and random.random() < 0.45:
            article, obj, verb, subject, punct = passive_match.groups()
            if not (verb.endswith("ed") or verb.lower() in IRREGULAR_PARTICIPLES):
                transformed.append(s)
                continue
            active_verb = IRREGULAR_PARTICIPLES.get(verb.lower(), verb)
            punct = punct or "."
            transformed.append(f"{subject} {active_verb} {article.lower()} {obj}{punct}")
            continue

        # Active -> Passive: "Alice completed the report." -> "The report was completed by Alice."
        active_match = re.match(
            r"^([A-Z][A-Za-z'-]+)\s+([A-Za-z]+ed)\s+(the|a|an)\s+(.+?)([.!?]?)$",
            s,
        )
        if active_match and random.random() < 0.2:
            subject, verb, article, obj, punct = active_match.groups()
            punct = punct or "."
            transformed.append(f"{article.capitalize()} {obj} was {verb} by {subject}{punct}")
            continue

        transformed.append(s)

    return _join_sentences(transformed)

# Main function to apply all style transformations to the input text, including contractions, casual word replacements, hedging, fillers, discourse markers, and openers. The transformations are applied in a specific order to ensure a natural flow of the text.
def restructure_sentences(text: str) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return text

    restructured: List[str] = []
    for sentence in sentences:
        s = sentence.strip()
        words = s.split()

        # Split long compound sentences into two shorter ones.
        if len(words) >= 18:
            split_match = re.match(r"^(.+?),\s+(and|but|so|because)\s+(.+?)([.!?]?)$", s, flags=re.IGNORECASE)
            if split_match and random.random() < 0.45:
                left, connector, right, punct = split_match.groups()
                punct = punct or "."
                restructured.append(f"{left.strip()}. {connector.capitalize()} {right.strip()}{punct}")
                continue

        # Reorder a sentence around "because" to vary structure.
        because_match = re.match(r"^(.+?)\s+because\s+(.+?)([.!?]?)$", s, flags=re.IGNORECASE)
        if because_match and random.random() < 0.25:
            main_clause, reason, punct = because_match.groups()
            punct = punct or "."
            reason_clean = reason.strip()
            main_clean = main_clause.strip()
            if len(main_clean) > 1:
                restructured.append(f"Because {reason_clean}, {main_clean[0].lower()}{main_clean[1:]}{punct}")
            else:
                restructured.append(f"Because {reason_clean}, {main_clean}{punct}")
            continue

        restructured.append(s)

    return _join_sentences(restructured)

# Main function to apply all style transformations to the input text, including contractions, casual word replacements, hedging, fillers, discourse markers, and openers. The transformations are applied in a specific order to ensure a natural flow of the text.
def syntax_transform_spacy(text: str) -> str:
    nlp = _get_spacy_nlp()
    if nlp is None:
        return text

    try:
        doc = nlp(text)
    except Exception:
        return text

    transformed: List[str] = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        if len(sent) < 10 or random.random() >= 0.4:
            transformed.append(sent_text)
            continue

        root = next((t for t in sent if t.head == t), None)
        advcl = next((t for t in sent if root is not None and t.dep_ == "advcl" and t.head == root), None)
        if advcl is None:
            transformed.append(sent_text)
            continue

        clause_tokens = sorted(list(advcl.subtree), key=lambda t: t.i)
        clause_set = {t.i for t in clause_tokens}
        clause = " ".join(t.text for t in clause_tokens).strip(" ,")
        main = " ".join(t.text for t in sent if t.i not in clause_set).strip(" ,")

        if not clause or not main:
            transformed.append(sent_text)
            continue

        clause_lower = clause.lower()
        if not clause_lower.startswith(("because", "although", "when", "if", "while", "since")):
            clause = f"because {clause[0].lower()}{clause[1:]}" if len(clause) > 1 else f"because {clause}"

        punct = "." if re.search(r"[.!?]$", sent_text) is None else sent_text[-1]
        main_text = f"{main[0].lower()}{main[1:]}" if len(main) > 1 else main
        transformed.append(f"{clause[0].upper()}{clause[1:]}, {main_text}{punct}")

    return _join_sentences(transformed)

# Main function to apply all style transformations to the input text, including contractions, casual word replacements, hedging, fillers, discourse markers, and openers. The transformations are applied in a specific order to ensure a natural flow of the text.
def apply_hedging(text: str) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return text
    if random.random() < 0.45:
        hedge = random.choice(HEDGES)
        first = sentences[0]
        if not re.match(r"^(I think|maybe|in my view|it seems)\b", first, re.IGNORECASE):
            if hedge.lower() == "maybe":
                sentences[0] = f"Maybe {first}"
            else:
                sentences[0] = f"{hedge}, {first}"
    return _join_sentences(sentences)

# Main function to apply all style transformations to the input text, including contractions, casual word replacements, hedging, fillers, discourse markers, and openers. The transformations are applied in a specific order to ensure a natural flow of the text.
def add_filler(text: str) -> str:
    sentences = _split_sentences(text)
    if len(sentences) < 2 or random.random() >= 0.35:
        return text
    idx = random.randint(1, len(sentences) - 1)
    sentences[idx] = f"{random.choice(FILLERS).capitalize()}, {_lower_start_if_safe(sentences[idx])}"
    return _join_sentences(sentences)

# Main function to apply all style transformations to the input text, including contractions, casual word replacements, hedging, fillers, discourse markers, and openers. The transformations are applied in a specific order to ensure a natural flow of the text.
def add_discourse_marker(text: str) -> str:
    sentences = _split_sentences(text)
    if len(sentences) < 2 or random.random() >= 0.5:
        return text
    idx = random.randint(1, len(sentences) - 1)
    marker = random.choice(DISCOURSE_MARKERS)
    if not re.match(r"^(Also|Anyway|Plus|On top of that)\b", sentences[idx], re.IGNORECASE):
        sentences[idx] = f"{marker}, {_lower_start_if_safe(sentences[idx])}"
    return _join_sentences(sentences)

# Main function to apply all style transformations to the input text, including contractions, casual word replacements, hedging, fillers, discourse markers, and openers. The transformations are applied in a specific order to ensure a natural flow of the text.
def add_opener(text: str) -> str:
    if random.random() >= 0.3:
        return text
    sentences = _split_sentences(text)
    if not sentences:
        return text
    if re.match(r"^(Honestly|To be fair|In practice|At the end of the day)\b", sentences[0], re.IGNORECASE):
        return text
    sentences[0] = f"{random.choice(OPENERS)} {sentences[0]}"
    return _join_sentences(sentences)

# Main function to apply all style transformations to the input text, including contractions, casual word replacements, hedging, fillers, discourse markers, and openers. The transformations are applied in a specific order to ensure a natural flow of the text.
def merge_sentences(text: str) -> str:
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return text
# Iterate through sentences and randomly merge adjacent sentences with a certain probability if both sentences are relatively short. When merging, remove terminal punctuation from the first sentence and lowercase the start of the second sentence to create a more natural flow. Ensure that the merged sentence is properly spaced and punctuated.
    merged: List[str] = []
    i = 0
    while i < len(sentences):
        if i < len(sentences) - 1:
            left = sentences[i]
            right = sentences[i + 1]
            if len(left.split()) <= 10 and len(right.split()) <= 10 and random.random() < 0.35:
                left_clean = re.sub(r"[.!?]+$", "", left).strip()
                right_clean = right.strip()
                if re.match(r"^(And|But|So|Because|Also|Anyway|Plus|On top of that)\b", right_clean, re.IGNORECASE):
                    merged.append(f"{left_clean}, {right_clean}")
                else:
                    merged.append(f"{left_clean}, and {right_clean}")
                i += 2
                continue
        merged.append(sentences[i])
        i += 1

    return _join_sentences(merged)

# Main function to apply all style transformations to the input text, including contractions, casual word replacements, hedging, fillers, discourse markers, and openers. The transformations are applied in a specific order to ensure a natural flow of the text.
def add_connectors(text: str) -> str:
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return text

    idx = random.randint(1, len(sentences) - 1)
    if random.random() < 0.5:
        connector = random.choice(CONNECTORS).capitalize()
        target = sentences[idx]
        if not re.match(r"^(And|But|So|Because)\b", target, re.IGNORECASE):
            sentences[idx] = f"{connector}, {_lower_start_if_safe(target)}"
    return _join_sentences(sentences)

# Main function to apply all style transformations to the input text, including contractions, casual word replacements, hedging, fillers, discourse markers, and openers. The transformations are applied in a specific order to ensure a natural flow of the text.
def style_transform(text: str) -> str:
    text = apply_contractions(text)
    text = casualize(text)
    text = transform_voice(text)
    text = restructure_sentences(text)
    text = syntax_transform_spacy(text)
    text = add_connectors(text)
    text = apply_hedging(text)
    text = add_filler(text)
    text = add_discourse_marker(text)
    text = add_opener(text)
    text = merge_sentences(text)
    return text

# Public function to apply style transformation to the input text, with error handling to ensure that any issues during the transformation process do not cause the application to fail and instead return the original text.
def apply_style(text: str) -> str:
    if not text:
        return text
    return style_transform(text)
