import random


def inject_human_noise(text: str) -> str:
    if not text:
        return text

    if random.random() < 0.25:
        replacements = {
            " do not ": " don't ",
            " cannot ": " can't ",
            "I am": "I'm",
            "it is": "it's",
        }
        noisy = f" {text} "
        for src, dst in replacements.items():
            noisy = noisy.replace(src, dst)
        return noisy.strip()
    return text
