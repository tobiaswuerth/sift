import pytest

from tools.check_crate import LAYERS, MOST_LINES, crate, main


def problems(sources: dict[str, str]) -> list[tuple[str, int, str]]:
    return [(finding.path, finding.line, finding.problem) for finding in crate(sources)]


def test_a_module_longer_than_the_most_lines_is_found() -> None:
    sources = {
        "native/src/numbers.rs": "const A: u8 = 1;\n" * (MOST_LINES + 1),
        "native/src/packed.rs": "const A: u8 = 1;\n" * MOST_LINES,
    }
    assert problems(sources) == [
        ("native/src/numbers.rs", 1, f"module of {MOST_LINES + 1} lines, more than {MOST_LINES}")
    ]


def test_a_type_named_twice_in_the_crate_is_found_where_it_is_named_again() -> None:
    sources = {
        "native/src/lists.rs": "pub struct Row {\n}\n\ntype Ids = Vec<i64>;\npub(crate) trait Told {}\n",
        "native/src/merge/sources.rs": "enum Row {\n}\n\nstruct Source;\n",
        "native/src/tests/support.rs": "    pub type Ids = u8;\nfn told() {}\n",
    }
    assert problems(sources) == [
        ("native/src/merge/sources.rs", 1, "Row is defined in native/src/lists.rs:1 too"),
        ("native/src/tests/support.rs", 1, "Ids is defined in native/src/lists.rs:4 too"),
    ]


def test_a_module_that_uses_one_above_it_or_lies_in_no_layer_is_found_and_the_tests_use_any() -> None:
    sources = {
        "native/src/lib.rs": "mod lists;\nmod merge;\n",
        "native/src/lists.rs": "use crate::numbers::read;\nuse crate::merge::Merger;\n",
        "native/src/merge/joining.rs": "use crate::merge::sources::Fed;\nfn f() { crate::python::listed(); }\n",
        "native/src/python/lists.rs": "use crate::lists::blocks;\nuse crate::python::numbers_of;\n",
        "native/src/tests/merge.rs": "use crate::python::listed;\nuse crate::tests::support::table;\n",
        "native/src/extra.rs": "use crate::numbers::read;\n",
    }
    assert problems(sources) == [
        ("native/src/extra.rs", 1, "extra is in no layer of the crate"),
        ("native/src/lists.rs", 2, "lists uses merge, a layer above it"),
        ("native/src/merge/joining.rs", 2, "merge uses python, a layer above it"),
    ]
    assert LAYERS[-1] == "python"


def test_a_comment_is_found_and_slashes_inside_text_are_none() -> None:
    source = r"""const A: &str = "http://x // none \" // none";
const B: &str = r#"/* raw " // none */"#;
const C: char = '/'; // said
/* a block */
fn held<'a>(text: &'a str, quote: char) -> bool { quote == '"' || quote == '\'' || text.len() / 2 > 1 }
/// told
"""
    assert problems({"native/src/words.rs": source}) == [
        ("native/src/words.rs", 3, "holds a comment"),
        ("native/src/words.rs", 4, "holds a comment"),
        ("native/src/words.rs", 6, "holds a comment"),
    ]


def test_the_crate_keeps_its_shape(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert capsys.readouterr().out == ""
