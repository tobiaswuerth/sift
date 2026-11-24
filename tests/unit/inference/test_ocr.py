import io
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import torch
from PIL import Image

from sift.api.errors import ModelUnavailableError
from sift.domain.text.scans import LOOP_CHARACTERS, SHORTEST
from sift.inference.accelerator import CPU, Device
from sift.inference.hub import OCR_MODEL, PINNED
from sift.inference.loading import ocr_module
from sift.inference.ocr import (
    LOOP_STEP,
    NEW_TOKENS,
    RETRY_PENALTY,
    Ocr,
    loaded_ocr,
    ocr_device,
    page_text,
    pages_text,
    stops,
)
from sift.ingest.pages import pages
from sift.ingest.pictures import Pages
from tests.scans import scan

if TYPE_CHECKING:
    from transformers import StoppingCriteriaList

GPU = Device("cuda", torch.bfloat16)
LOOPED = 7
AFTER_THE_LOOP = (8, 9)


@dataclass(slots=True)
class FakeProcessor:
    def apply_chat_template(
        self, conversation: Sequence[Mapping[str, object]], **options: object
    ) -> Mapping[str, torch.Tensor]:
        assert options == {"add_generation_prompt": True, "tokenize": True, "return_dict": True, "return_tensors": "pt"}
        (part,) = cast("list[dict[str, Image.Image]]", conversation[0]["content"])
        picture = part["image"]
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "pixel_values": torch.zeros(1, 1),
            "image_sizes": torch.tensor([[picture.height, picture.width]]),
        }

    def decode(self, ids: torch.Tensor, /, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return " ".join(f"word{int(each)}" for each in ids if int(each))


@dataclass(slots=True)
class FakeModel:
    loop: bool = False
    loop_again: bool = False
    full: bool = False
    device: str = "cpu"
    generated: list[int] = field(default_factory=list[int])
    penalties: list[float | None] = field(default_factory=list[float | None])
    working: list["weakref.ref[torch.Tensor]"] = field(default_factory=list["weakref.ref[torch.Tensor]"])

    def to(self, device: str, /) -> "FakeModel":
        self.device = device
        return self

    def eval(self) -> "FakeModel":
        return self

    def generate(self, **inputs: object) -> torch.Tensor:
        assert (inputs["do_sample"], inputs["max_new_tokens"]) == (False, NEW_TOKENS)
        if self.full:
            held = torch.zeros(4)
            self.working.append(weakref.ref(held))
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 3.00 GiB")
        prompt = cast("torch.Tensor", inputs["input_ids"])
        penalty = cast("float | None", inputs.get("repetition_penalty"))
        stops = cast("StoppingCriteriaList", inputs["stopping_criteria"])
        self.penalties.append(penalty)
        height, width = (int(each) for each in cast("torch.Tensor", inputs["image_sizes"])[0])
        ids = [int(each) for each in prompt[0]] + [width, height]
        looping = self.loop and (penalty is None or self.loop_again)
        while looping and len(ids) - prompt.shape[1] < NEW_TOKENS:
            ids.append(LOOPED)
            if bool(stops(torch.tensor([ids]), None).all()):
                break
        if self.loop and not looping:
            ids += AFTER_THE_LOOP
        self.generated.append(len(ids) - prompt.shape[1])
        return torch.tensor([[*ids, 0]])


def ocr_of(model: FakeModel) -> Ocr:
    return Ocr(model, FakeProcessor(), CPU)


def a4() -> Image.Image:
    found = pages(scan())
    assert isinstance(found, Pages)
    return Image.open(io.BytesIO(found.pictures[0]))


def test_a_page_is_read_as_the_words_the_model_makes_of_it() -> None:
    found = pages(scan(pages=2))
    assert isinstance(found, Pages)
    model = FakeModel()
    assert pages_text(ocr_of(model), found.pictures) == ["word1089 word1540"] * 2
    assert model.penalties == [None, None]


def test_a_page_that_loops_is_read_again_with_a_repetition_penalty() -> None:
    model = FakeModel(loop=True)
    assert page_text(ocr_of(model), a4()) == "word1089 word1540 word8 word9"
    assert model.penalties == [None, RETRY_PENALTY]
    assert model.generated[0] <= 2 + LOOP_CHARACTERS // len(" word7") + LOOP_STEP


def _thin() -> Image.Image:
    picture = Image.new("RGB", (SHORTEST - 1, 1540), "white")
    picture.paste("black", (0, 100, 1, 200))
    return picture


@pytest.mark.parametrize("picture", [Image.new("RGB", (1089, 1540), "white"), _thin()], ids=["blank", "thin"])
def test_a_page_of_one_colour_or_too_thin_for_the_model_is_no_text_the_model_is_not_asked(picture: Image.Image) -> None:
    model = FakeModel()
    assert (page_text(ocr_of(model), picture), model.generated) == ("", [])


def test_a_loop_the_second_reading_does_not_better_is_cut() -> None:
    assert page_text(ocr_of(FakeModel(loop=True, loop_again=True)), a4()) == "word1089 word1540 word7"


def test_a_model_out_of_memory_on_a_page_is_unavailable_saying_so_its_device_emptied_for_the_next() -> None:
    found = pages(scan())
    assert isinstance(found, Pages)
    emptied: list[bool] = []
    full = FakeModel(full=True)
    with pytest.raises(ModelUnavailableError, match="ran out of memory on the cpu"):
        pages_text(ocr_of(full), found.pictures, {"cpu": lambda: emptied.append(full.working[0]() is None)})
    assert emptied == [True]
    assert pages_text(ocr_of(FakeModel()), found.pictures, {"cpu": lambda: emptied.append(False)}) == [
        "word1089 word1540"
    ]
    assert emptied == [True]


def test_the_model_loads_at_its_pinned_revision_onto_its_device_in_its_precision() -> None:
    model, asked = FakeModel(), list[tuple[str, object, str]]()

    def load_model(repository: str, /, *, dtype: torch.dtype, revision: str) -> object:
        asked.append((repository, dtype, revision))
        return model

    def load_processor(repository: str, /, *, revision: str) -> object:
        asked.append((repository, None, revision))
        return FakeProcessor()

    found = loaded_ocr(GPU, OCR_MODEL, load_model, load_processor)
    pinned = PINNED[OCR_MODEL]
    assert asked == [(OCR_MODEL, torch.bfloat16, pinned), (OCR_MODEL, None, pinned)]
    assert (found.model, model.device, found.device) == (model, "cuda", GPU)


def test_a_model_that_cannot_be_loaded_says_what_to_check() -> None:
    def failing(repository: str, /, *, dtype: torch.dtype, revision: str) -> object:
        raise OSError(f"401 Client Error: {repository}@{revision} in {dtype}")

    with pytest.raises(ModelUnavailableError, match=f"the model {OCR_MODEL} could not be loaded"):
        loaded_ocr(GPU, OCR_MODEL, failing)


def test_a_model_from_a_folder_that_holds_none_cannot_be_loaded(tmp_path: Path) -> None:
    with pytest.raises(ModelUnavailableError, match="could not be loaded"):
        loaded_ocr(CPU, str(tmp_path / "missing"))

    def model_alone(repository: str, /, *, dtype: torch.dtype, revision: str) -> object:
        assert (repository, dtype, revision) == (str(tmp_path / "missing"), CPU.dtype, "main")
        return FakeModel()

    with pytest.raises(ModelUnavailableError, match="could not be loaded"):
        loaded_ocr(CPU, str(tmp_path / "missing"), model_alone)


def test_ocr_runs_on_the_device_the_other_models_would_and_loads_when_first_needed() -> None:
    assert ocr_device(None).name in {"cpu", "cuda", "mps"}
    assert ocr_device("-1").name == "cpu"
    module = ocr_module()
    assert (module.ocr_device, module.loaded_ocr, module.pages_text) == (ocr_device, loaded_ocr, pages_text)


def test_a_page_is_told_to_stop_on_the_device_it_is_read_on() -> None:
    ids = cast("torch.LongTensor", torch.tensor([[1, 2, 3, 4]], device="meta"))
    assert stops(FakeProcessor(), 3)(ids, None).device == ids.device
