def optimize_perplexity(text: str) -> str:
    if not text:
        return text
    return " ".join(text.split())
