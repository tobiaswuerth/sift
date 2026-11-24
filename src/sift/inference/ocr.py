import io
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast, final, override

import torch
from PIL import Image
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor, StoppingCriteria, StoppingCriteriaList
from transformers.utils import logging as transformers_logging

from sift.api.errors import ModelUnavailableError
from sift.domain.text.scans import looping, without_loop, worth_reading
from sift.inference.accelerator import CACHES, Device, emptied
from sift.inference.hub import OCR_MODEL, hub
from sift.inference.local import loading, this_device, within_memory

NEW_TOKENS = 4096
LOOP_STEP = 16
RETRY_PENALTY = 1.1
COLOURS = "RGB"
GREY = "L"
DEFAULT_REVISION = "main"

type Inputs = Mapping[str, torch.Tensor]


class Generating(Protocol):
    def to(self, device: str, /) -> "Generating": ...

    def eval(self) -> "Generating": ...

    def generate(self, **inputs: object) -> torch.Tensor: ...


class Processing(Protocol):
    def apply_chat_template(self, conversation: Sequence[Mapping[str, object]], **options: object) -> Inputs: ...

    def decode(self, ids: torch.Tensor, /, *, skip_special_tokens: bool) -> str: ...


class LoadModel(Protocol):
    def __call__(self, repository: str, /, *, dtype: torch.dtype, revision: str) -> object: ...


class LoadProcessor(Protocol):
    def __call__(self, repository: str, /, *, revision: str) -> object: ...


class _PretrainedProcessor(Protocol):
    def __call__(self, repository: str, /, *, revision: str, fix_mistral_regex: bool) -> object: ...


@dataclass(frozen=True, slots=True)
class Ocr:
    model: Generating
    processor: Processing
    device: Device


@final
class _Looped(StoppingCriteria):
    def __init__(self, processor: Processing, prompt: int) -> None:
        self.processor, self.prompt = processor, prompt

    def _looped(self, input_ids: torch.LongTensor) -> bool:
        new = input_ids.shape[1] - self.prompt
        return new % LOOP_STEP == 0 and looping(
            self.processor.decode(input_ids[0, self.prompt :], skip_special_tokens=True)
        )

    @override
    def __call__(
        self, input_ids: torch.LongTensor, scores: tuple[torch.Tensor] | None, **kwargs: object
    ) -> torch.BoolTensor:
        stop = self._looped(input_ids)
        found = torch.full((input_ids.shape[0],), stop, dtype=torch.bool, device=input_ids.device)
        return cast("torch.BoolTensor", found)


def stops(processor: Processing, prompt: int) -> StoppingCriteriaList:
    return StoppingCriteriaList([_Looped(processor, prompt)])


def _moved(inputs: Inputs, device: Device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device.name, device.dtype) if value.is_floating_point() else value.to(device.name)
        for key, value in inputs.items()
    }


def _generated(ocr: Ocr, inputs: dict[str, torch.Tensor], penalty: float | None) -> str:
    prompt = inputs["input_ids"].shape[1]
    penalised = {} if penalty is None else {"repetition_penalty": penalty}
    with torch.inference_mode():
        made = ocr.model.generate(
            **inputs,
            max_new_tokens=NEW_TOKENS,
            do_sample=False,
            stopping_criteria=stops(ocr.processor, prompt),
            **penalised,
        )
    return ocr.processor.decode(made[0, prompt:], skip_special_tokens=True)


def _worth_reading(picture: Image.Image) -> bool:
    darkest, lightest = cast("tuple[int, int]", picture.convert(GREY).getextrema())
    return worth_reading(picture.width, picture.height, darkest, lightest)


def page_text(ocr: Ocr, picture: Image.Image) -> str:
    if not _worth_reading(picture):
        return ""
    asked = [{"role": "user", "content": [{"type": "image", "image": picture}]}]
    inputs = ocr.processor.apply_chat_template(
        asked, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    moved = _moved(inputs, ocr.device)
    text = _generated(ocr, moved, None)
    if looping(text):
        text = max(without_loop(text), without_loop(_generated(ocr, moved, RETRY_PENALTY)), key=len)
    return without_loop(text)


def _read(ocr: Ocr, picture: bytes) -> str:
    with Image.open(io.BytesIO(picture)) as opened:
        return page_text(ocr, opened.convert(COLOURS))


def _within_memory(ocr: Ocr, picture: bytes, caches: Mapping[str, Callable[[], None]]) -> str:
    try:
        return within_memory(OCR_MODEL, ocr.device.name, lambda: _read(ocr, picture))
    except ModelUnavailableError as error:
        said = error.message, error.remedy
    emptied(ocr.device, caches)
    raise ModelUnavailableError(*said)


def pages_text(ocr: Ocr, pictures: Iterable[bytes], caches: Mapping[str, Callable[[], None]] = CACHES) -> list[str]:
    return [_within_memory(ocr, picture, caches) for picture in pictures]


def _pretrained_model(repository: str, /, *, dtype: torch.dtype, revision: str) -> object:
    load = cast("LoadModel", LightOnOcrForConditionalGeneration.from_pretrained)
    return load(repository, dtype=dtype, revision=revision)


def _pretrained_processor(repository: str, /, *, revision: str) -> object:
    load = cast("_PretrainedProcessor", LightOnOcrProcessor.from_pretrained)
    return load(repository, revision=revision, fix_mistral_regex=True)


def loaded_ocr(
    device: Device,
    name: str = OCR_MODEL,
    load_model: LoadModel = _pretrained_model,
    load_processor: LoadProcessor = _pretrained_processor,
) -> Ocr:
    repository, pinned_at = hub(name)
    revision = pinned_at or DEFAULT_REVISION

    def load() -> Ocr:
        transformers_logging.set_verbosity_error()
        model = cast("Generating", load_model(repository, dtype=device.dtype, revision=revision))
        processor = cast("Processing", load_processor(repository, revision=revision))
        return Ocr(model.to(device.name).eval(), processor, device)

    return loading(name, device, load)


def ocr_device(gpus: str | None) -> Device:
    return this_device(gpus)
