from pathlib import Path
from typing import Protocol, cast

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Pooling, StaticEmbedding, Transformer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertForSequenceClassification, BertModel, BertTokenizer

WORDS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "dark", "matter", "halo", "models", "pruning", "skips", "ranges")
WIDTH = 16
POSITIONS = 64
VOCABULARY = {word: at for at, word in enumerate(WORDS)}


class Saved(Protocol):
    def save_pretrained(self, folder: Path) -> object: ...


def saved(made: object) -> Saved:
    return cast("Saved", made)


def _config() -> BertConfig:
    return BertConfig(
        vocab_size=len(WORDS),
        hidden_size=WIDTH,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=POSITIONS,
    )


def _vocabulary(folder: Path, longest: int = POSITIONS) -> Path:
    folder.mkdir(parents=True)
    saved(BertTokenizer(vocab=VOCABULARY, model_max_length=longest)).save_pretrained(folder)
    return folder


def tiny_reranker(folder: Path) -> Path:
    made = _vocabulary(folder / "reranker")
    config = _config()
    config.num_labels = 1
    saved(BertForSequenceClassification(config)).save_pretrained(made)
    return made


def tiny_model(folder: Path) -> Path:
    raw, made = _vocabulary(folder / "raw"), folder / "tiny"
    saved(BertModel(_config())).save_pretrained(raw)
    SentenceTransformer(modules=[Transformer(str(raw), max_seq_length=32), Pooling(WIDTH)]).save(str(made))
    return made


def tiny_static(folder: Path) -> Path:
    tokenizer = Tokenizer(WordLevel(VOCABULARY, "[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    made = folder / "static"
    SentenceTransformer(modules=[StaticEmbedding(tokenizer, embedding_dim=WIDTH)]).save(str(made))
    return made


def tiny_prompted(folder: Path, positions: int) -> Path:
    raw, made = _vocabulary(folder / f"prompted-raw-{positions}", positions), folder / f"prompted-{positions}"
    config = _config()
    config.max_position_embeddings = positions
    saved(BertModel(config)).save_pretrained(raw)
    model = SentenceTransformer(
        modules=[Transformer(str(raw), max_seq_length=positions), Pooling(WIDTH)],
        prompts={"query": "halo ", "document": "pruning "},
    )
    model.save(str(made))
    return made
