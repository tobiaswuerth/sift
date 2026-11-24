from types import SimpleNamespace

import pytest

from tools import check_stubs
from tools.check_stubs import missing, module_of


class Bitmap:
    def to_pil(self) -> None: ...


class FailedError(RuntimeError): ...


PACKAGE = SimpleNamespace(LIMIT=1, Bitmap=Bitmap, FailedError=FailedError, Page=object(), opened=print)


def found(stub: str) -> list[tuple[int, str]]:
    return [(each.line, each.problem) for each in missing({"typings/pdf.pyi": stub}, {"pdf": PACKAGE}.__getitem__)]


def test_a_stub_that_names_what_its_package_has_passes() -> None:
    stub = (
        "from PIL.Image import Image\n\nLIMIT: int\n\nclass FailedError(RuntimeError): ...\n\n"
        "class Bitmap:\n    def to_pil(self) -> Image: ...\n\ndef opened() -> None: ...\n"
    )
    assert found(stub) == []


def test_a_name_the_package_lacks_is_found_at_the_top_and_in_a_class() -> None:
    stub = (
        "SIZE: int\nGONE = 1\n\nclass Bitmap:\n    width: int\n"
        "    def to_pil(self) -> None: ...\n    def close(self) -> None: ...\n"
    )
    assert found(stub) == [
        (1, "names pdf.SIZE, which the installed package lacks"),
        (2, "names pdf.GONE, which the installed package lacks"),
        (5, "names pdf.Bitmap.width, which the installed package lacks"),
        (7, "names pdf.Bitmap.close, which the installed package lacks"),
    ]


def test_a_class_must_be_a_class_and_derive_from_the_builtins_it_names() -> None:
    stub = (
        "class Page: ...\n\nclass Bitmap(ValueError, Image): ...\n\nclass FailedError(RuntimeError, Exception): ...\n"
    )
    assert found(stub) == [
        (1, "names the class pdf.Page, which is no class"),
        (3, "says pdf.Bitmap is a ValueError, which it is not"),
    ]


@pytest.mark.parametrize(
    "named",
    [("typings/pynvml.pyi", "pynvml"), ("typings/pypdfium2/__init__.pyi", "pypdfium2"), ("typings/a/b.pyi", "a.b")],
)
def test_a_stub_is_of_the_module_its_path_names(named: tuple[str, str]) -> None:
    path, module = named
    assert module_of(path) == module


def test_the_stubs_name_what_their_packages_have() -> None:
    assert check_stubs.main([]) == 0
