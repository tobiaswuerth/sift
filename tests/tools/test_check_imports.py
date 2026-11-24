from tools import check_imports
from tools.check_imports import declared, undeclared

PROVIDERS = {"numpy": ["numpy"], "huggingface_hub": ["huggingface-hub"], "PIL": ["Pillow"], "pynvml": ["nvidia-ml-py"]}


def found(sources: dict[str, str], depended: frozenset[str]) -> list[tuple[str, int, str]]:
    return [(each.path, each.line, each.problem) for each in undeclared(sources, PROVIDERS, depended)]


def test_a_package_imported_that_pyproject_does_not_depend_on_is_found() -> None:
    sources = {
        "src/sift/a.py": "import json\nimport numpy as np\nfrom huggingface_hub import utils\nfrom sift.b import y\n",
        "src/sift/b.py": "from __future__ import annotations\n\nimport PIL.Image\nimport pynvml\nfrom . import a\n",
    }
    assert found(sources, frozenset({"numpy", "pillow", "nvidia-ml-py"})) == [
        ("src/sift/a.py", 3, "imports huggingface_hub, of huggingface-hub, which pyproject.toml does not depend on"),
    ]


def test_a_package_no_installed_distribution_provides_is_found() -> None:
    assert found({"src/sift/a.py": "import nowhere.deep\n"}, frozenset()) == [
        ("src/sift/a.py", 1, "imports nowhere, which no installed distribution provides"),
    ]


def test_dependencies_are_known_by_their_normalised_names_whatever_follows_them() -> None:
    pyproject = (
        '[project]\ndependencies = ["Pillow>=13", "nvidia_ml.py>=12  ", "torch>=2.13; sys_platform == \'linux\'",'
        ' "xberg[pdf]>=1.3"]\n'
    )
    assert declared(pyproject) == frozenset({"pillow", "nvidia-ml-py", "torch", "xberg"})


def test_every_package_sift_imports_is_a_dependency() -> None:
    assert check_imports.main([]) == 0
