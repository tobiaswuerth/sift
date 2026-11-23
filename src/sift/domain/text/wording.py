def counted(n: int, noun: str) -> str:
    return f"{n:,} {noun}{'' if n == 1 else 's'}"
