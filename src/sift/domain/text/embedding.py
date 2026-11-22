def embedded(title: str, text: str, *, first: bool) -> str:
    return text if first or text.lstrip("# ").startswith(title) else f"{title}\n\n{text}"
