import importlib.metadata

import pytest

from sift.inference.pairing import Release, read_unpaired, unpaired

TORCH = Release("torch", "2.14.0+cu130")
BUILT_FOR = ["sympy>=1.13.3", 'triton~=3.8.0; platform_system == "Linux" and python_version < "3.15"']
OTHER = Release("triton-windows", "3.9.0.post30")
HINT = (
    "PyTorch 2.14.0+cu130 is built for Triton 3.8, but triton-windows 3.9.0.post30 is installed: with uv, "
    'uv tool install --reinstall --torch-backend auto --with "triton-windows~=3.8.0" sift-rag; with pip, '
    'pip install "triton-windows~=3.8.0"'
)


def test_a_triton_of_another_minor_release_than_pytorch_was_built_for_is_named_with_the_one_to_install() -> None:
    assert unpaired(TORCH, BUILT_FOR, OTHER) == (HINT,)
    assert unpaired(TORCH, ["triton==3.1.0"], Release("triton", "3.10.0"))[0].startswith(
        "PyTorch 2.14.0+cu130 is built for Triton 3.1, but triton 3.10.0 is installed"
    )


@pytest.mark.parametrize(
    "requires, triton",
    [
        (BUILT_FOR, Release("triton-windows", "3.8.0.post29")),
        (['triton==3.8.1; platform_system == "Linux"'], Release("triton", "3.8.1")),
        (["sympy>=1.13.3", "triton-windows~=3.1.0", "triton>=3.0"], OTHER),
        ([], OTHER),
        (BUILT_FOR, None),
        (BUILT_FOR, Release("triton", "nightly")),
    ],
)
def test_no_hint_is_given_when_triton_matches_or_either_release_is_unknown(
    requires: list[str], triton: Release | None
) -> None:
    assert unpaired(TORCH, requires, triton) == ()


def missing(name: str) -> str:
    raise importlib.metadata.PackageNotFoundError(name)


def unrequired(_: str) -> list[str] | None:
    return None


def test_the_releases_installed_are_read_from_the_packages_metadata() -> None:
    held = {"torch": "2.14.0+cu130", "triton-windows": "3.9.0.post30"}
    providers = {"triton": ["triton-windows"], "numpy": ["numpy"]}
    assert read_unpaired(lambda: providers, held.__getitem__, {"torch": BUILT_FOR}.get) == (HINT,)
    assert read_unpaired(lambda: {"numpy": ["numpy"]}, held.__getitem__, {"torch": BUILT_FOR}.get) == ()
    assert read_unpaired(lambda: providers, held.__getitem__, unrequired) == ()
    assert read_unpaired(lambda: providers, missing, {"torch": BUILT_FOR}.get) == ()


def test_the_metadata_of_this_installation_gives_at_most_one_hint() -> None:
    found = read_unpaired()
    assert (len(found) <= 1, all(isinstance(each, str) for each in found)) == (True, True)
