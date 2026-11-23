from pathlib import Path
from typing import Protocol, cast

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
from transformers import BertConfig, BertModel, BertTokenizer

WORDS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "dark", "matter", "halo", "models", "pruning", "skips", "ranges")
WIDTH = 16


class Saved(Protocol):
    def save_pretrained(self, folder: Path) -> object: ...


def saved(made: object) -> Saved:
    return cast("Saved", made)


def tiny_model(folder: Path) -> Path:
    raw, made = folder / "raw", folder / "tiny"
    raw.mkdir(parents=True)
    (raw / "vocab.txt").write_bytes("\n".join(WORDS).encode())
    config = BertConfig(
        vocab_size=len(WORDS),
        hidden_size=WIDTH,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    saved(BertModel(config)).save_pretrained(raw)
    saved(BertTokenizer(vocab_file=str(raw / "vocab.txt"))).save_pretrained(raw)
    SentenceTransformer(modules=[Transformer(str(raw), max_seq_length=32), Pooling(WIDTH)]).save(str(made))
    return made
