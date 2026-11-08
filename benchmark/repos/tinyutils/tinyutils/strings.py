"""String helpers."""


def truncate(text: str, max_len: int, suffix: str = "...") -> str:
    """Shorten text to at most max_len characters, adding suffix when cut."""
    if len(text) <= max_len:
        return text
    if max_len <= len(suffix):
        return suffix[:max_len]
    return text[: max_len - len(suffix)] + suffix


def is_palindrome(text: str) -> bool:
    cleaned = [c.lower() for c in text if c.isalnum()]
    return cleaned == cleaned[::-1]


def count_words(text: str) -> int:
    return len(text.split())


def capitalize_words(text: str) -> str:
    return " ".join(w[:1].upper() + w[1:].lower() for w in text.split(" "))


def slugify(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    return "".join(out).strip("-")


def first_word(text: str) -> str:
    words = text.split()
    return words[0] if words else ""
