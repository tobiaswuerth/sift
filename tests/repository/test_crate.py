import tomllib
from pathlib import Path
from typing import cast

ROOT = Path(__file__).parents[2]


def table(name: str) -> dict[str, object]:
    return tomllib.loads((ROOT / name).read_text(encoding="utf-8"))


def within(held: object, *keys: str) -> object:
    for key in keys:
        held = cast("dict[str, object]", held)[key]
    return held


def test_the_extension_s_allocator_keeps_its_thread_state_as_a_module_loaded_late_must() -> None:
    allocator = within(table("native/Cargo.toml"), "dependencies", "mimalloc")
    assert within(allocator, "features") == ["local_dynamic_tls"]
    assert (within(allocator, "default-features"), within(allocator, "optional")) == (False, True)


def test_a_build_takes_the_crates_the_lock_names_and_the_crate_says_the_oldest_rust_it_builds_with() -> None:
    assert within(table("pyproject.toml"), "tool", "maturin", "locked") is True
    pinned = str(within(table("rust-toolchain.toml"), "toolchain", "channel"))
    assert within(table("native/Cargo.toml"), "package", "rust-version") == pinned.rsplit(".", 1)[0]


def test_the_crate_s_coverage_is_refused_when_the_report_holds_nothing_of_the_binding() -> None:
    noxfile = (ROOT / "noxfile.py").read_text(encoding="utf-8")
    assert 'bound = [Path("python.rs"), *sorted(Path("python", each.name) for each in BINDING.glob("*.rs"))]' in noxfile
    assert "session.error(NO_BINDING.format(" in noxfile
