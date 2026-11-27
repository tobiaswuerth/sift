from pathlib import Path

import pytest

from tools.notices import NOTICES, RULE, Crate, NoticeError, linked, main, notices, texts

MIT = "Permission is hereby granted, free of charge.\n"


def package(name: str, folder: Path, *, macro: bool = False, licence: str | None = "MIT") -> dict[str, object]:
    return {
        "id": f"id-{name}",
        "name": name,
        "version": "1.2.0",
        "license": licence,
        "manifest_path": str(folder / name / "Cargo.toml"),
        "targets": [{"kind": ["proc-macro" if macro else "lib"]}],
    }


def node(name: str, **needs: tuple[str | None, ...]) -> dict[str, object]:
    deps = [{"pkg": f"id-{needed}", "dep_kinds": [{"kind": kind} for kind in kinds]} for needed, kinds in needs.items()]
    return {"id": f"id-{name}", "deps": deps}


def described(folder: Path) -> dict[str, object]:
    names = ("root", "a", "b", "built", "tried", "both", "macro", "behind")
    packages = [package(name, folder, macro=name == "macro", licence=None if name == "b" else "MIT") for name in names]
    nodes = [
        node("root", a=(None,), built=("build",), tried=("dev",), both=("build", None), macro=(None,)),
        node("a", b=(None,)),
        node("macro", behind=(None,)),
        *(node(name) for name in ("b", "built", "tried", "both", "behind")),
    ]
    return {"packages": packages, "resolve": {"root": "id-root", "nodes": nodes}}


def test_the_crates_linked_are_those_a_build_needs_at_run_time_and_no_macro_s() -> None:
    folder = Path("crates")
    assert linked(described(folder)) == [
        Crate("a", "1.2.0", "MIT", folder / "a"),
        Crate("b", "1.2.0", "", folder / "b"),
        Crate("both", "1.2.0", "MIT", folder / "both"),
    ]


def test_a_crate_s_licence_files_are_read_wherever_they_lie_tidied_and_nothing_else(tmp_path: Path) -> None:
    (tmp_path / "bundled" / "c").mkdir(parents=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "LICENSE-MIT").write_bytes(b"\r\n\r\nCopyright (c) Somebody  \r\n\r\nPermission.\t\r\n\r\n")
    (tmp_path / "license.txt").write_bytes(b"Another.\n")
    (tmp_path / "bundled" / "c" / "COPYING").write_bytes(b"Of the C sources.")
    for other in ("README.md", "src/license.rs", "COPYRIGHT.html", "Cargo.toml"):
        (tmp_path / other).write_bytes(b"no licence text\n")
    assert texts(Crate("a", "1.2.0", "MIT", tmp_path)) == [
        ("LICENSE-MIT", "Copyright (c) Somebody\n\nPermission.\n"),
        ("bundled/c/COPYING", "Of the C sources.\n"),
        ("license.txt", "Another.\n"),
    ]


def test_a_crate_without_a_licence_file_or_with_one_that_is_no_text_is_refused(tmp_path: Path) -> None:
    with pytest.raises(NoticeError, match=r"a 1\.2\.0 brings no licence file"):
        texts(Crate("a", "1.2.0", "MIT", tmp_path))
    (tmp_path / "LICENSE").write_bytes(b"\xff\xfe")
    with pytest.raises(NoticeError, match=r"LICENSE of a 1\.2\.0 is no UTF-8 text"):
        texts(Crate("a", "1.2.0", "MIT", tmp_path))


def test_each_text_stands_once_under_the_crates_that_carry_it() -> None:
    folder = Path("crates")
    alpha, beta = Crate("alpha", "1.0.0", "MIT OR Apache-2.0", folder), Crate("beta", "0.3.1", "", folder)
    said = notices([(beta, [("LICENSE", MIT)]), (alpha, [("LICENSE-MIT", MIT), ("LICENSE-APACHE", "Apache.\n")])])
    head, *parts = said.split(f"{RULE}\n")
    assert "sift._native" in head
    assert "Rust standard library" in head
    assert parts == [
        "alpha 1.0.0 (MIT OR Apache-2.0): LICENSE-APACHE\n",
        "\nApache.\n\n",
        "alpha 1.0.0 (MIT OR Apache-2.0): LICENSE-MIT\nbeta 0.3.1: LICENSE\n",
        f"\n{MIT}",
    ]


def test_the_notices_shipped_are_those_of_the_crates_the_lock_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (main([]), capsys.readouterr().err) == (0, "")
    stale = tmp_path / "notices"
    stale.write_bytes(b"of other crates\n")
    assert main([str(stale)]) == 1
    told = capsys.readouterr()
    assert (told.out, told.err.startswith(f"error  {stale} is not what the crates"), "--write" in told.err) == (
        "",
        True,
        True,
    )
    assert main(["--write", str(stale)]) == 0
    assert stale.read_bytes() == NOTICES.read_bytes()
    assert (main([str(stale)]), capsys.readouterr().err) == (0, "")
    assert (main([str(tmp_path / "missing")]), capsys.readouterr().err.startswith("error  ")) == (1, True)
    assert main(["--write"]) == 0
    assert (main(["--other"]), capsys.readouterr().err.startswith("usage: python -m tools notices")) == (2, True)
