import random


def apply_burstiness(text: str) -> str:
    if not text:
        return text

    parts = [p.strip() for p in text.split(".") if p.strip()]
    if len(parts) < 2:
        return text

    if random.random() < 0.35:
        i = random.randint(0, len(parts) - 1)
        parts[i] = parts[i] + " " + parts[i][: max(1, len(parts[i]) // 3)]
    return ". ".join(parts) + "."
