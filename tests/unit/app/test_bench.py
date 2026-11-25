import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from sift.api.bench import MeasureSearches, Mode, ModeMeasured
from sift.api.errors import InvalidRequestError
from sift.api.events import Event, Phase, PhaseDone, PhaseStarted
from sift.app.bench import MOST_FILE_BYTES, Case, cases_of, measured
from tests.homes import home_of
from tests.models import Hashed

NOTES = {
    "notes/rl.md": "# Reward shaping\n\nReward shaping speeds up reinforcement learning.\n",
    "notes/robots/brush.md": "# Brush\n\nA painting robot plans its brush strokes.\n",
    "notes/plain.txt": "Plain words about nothing in particular.\n",
}
CASES = [
    {"query": "reward shaping reinforcement", "relevant": ["notes/rl.md"]},
    {"query": "painting robot brush strokes", "relevant": {"notes/robots/brush.md": 2}},
]


@dataclass
class Ticking:
    now: float = 0.0
    read: list[float] = field(default_factory=list[float])

    def __call__(self) -> float:
        self.now += 0.5
        self.read.append(self.now)
        return self.now


def written(tmp_path: Path, cases: object, name: str = "cases.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(cases), encoding="utf-8")
    return path


@pytest.fixture
def models() -> Hashed:
    return Hashed()


@pytest.fixture
def home(tmp_path: Path, models: Hashed) -> Path:
    return home_of(tmp_path, NOTES, embedded=models)


def test_bench_measures_every_mode_on_every_question_and_times_each(home: Path, tmp_path: Path, models: Hashed) -> None:
    heard, clock = list[Event](), Ticking()
    result = measured(home, MeasureSearches(str(written(tmp_path, CASES))), models, heard.append, clock)
    assert (result.cases, [each.mode for each in result.modes]) == (2, [Mode.KEYWORD, Mode.SEMANTIC, Mode.HYBRID])
    assert result.modes == tuple(ModeMeasured(mode, 1.0, 1.0, 1.0, 250.0) for mode in Mode)
    measuring = [each for each in heard if isinstance(each, PhaseStarted) and each.phase is Phase.MEASURE]
    assert (measuring, heard[-1]) == ([PhaseStarted(Phase.MEASURE, 2)] * 3, PhaseDone(2))


def test_each_mode_embeds_each_question_itself_nothing_kept_from_another(
    home: Path, tmp_path: Path, models: Hashed
) -> None:
    before = len(models.asked)
    measured(home, MeasureSearches(str(written(tmp_path, CASES))), models, lambda _: None, Ticking())
    questions = [texts for texts, query in models.asked[before:] if query]
    assert len(questions) == 2 * (len(CASES) + 1)


def test_the_searches_measured_search_the_collections_named_alone(tmp_path: Path, models: Hashed) -> None:
    files = {**NOTES, "more/halo.md": "# Halo\n\nDark matter halo models.\n"}
    home = home_of(tmp_path / "two", files, embedded=models)
    asked = MeasureSearches(str(written(tmp_path, CASES)), collections=("more",))
    result = measured(home, asked, models, lambda _: None, Ticking())
    assert [(each.ndcg, each.recall) for each in result.modes] == [(0.0, 0.0)] * 3


def test_bench_takes_documents_as_get_does(home: Path, tmp_path: Path) -> None:
    named = [{"query": "reward", "relevant": [str(tmp_path / "notes" / "rl.md"), "notes/robots/brush.md"]}]
    assert cases_of(home, written(tmp_path, named)) == (
        Case("reward", {"notes/rl.md": 1.0, "notes/robots/brush.md": 1.0}),
    )


def test_documents_graded_0_are_not_relevant(home: Path, tmp_path: Path) -> None:
    graded = [{"query": "reward", "relevant": {"notes/rl.md": 1, "notes/plain.txt": 0}}]
    assert cases_of(home, written(tmp_path, graded)) == (Case("reward", {"notes/rl.md": 1.0}),)


RULED_OUT = "case 1 needs"
REFUSED = (
    ('[{"query": "x", "relevant": ["notes/missing.md"]}]', "case 1 names no note or document 'notes/missing.md'"),
    ('[{"query": "x", "relevant": []}]', RULED_OUT),
    ('[{"query": "x", "relevant": "notes/rl.md"}]', RULED_OUT),
    ('[{"query": "x", "relevant": [["notes/rl.md"]]}]', RULED_OUT),
    ('[{"query": "x", "relevant": ["notes/rl.md", 1]}]', RULED_OUT),
    ('[{"query": "x", "relevant": {"notes/rl.md": 1, "notes/plain.txt": "high"}}]', RULED_OUT),
    ('[{"query": "x", "relevant": {"notes/rl.md": 0}}]', RULED_OUT),
    ('[{"query": "x", "relevant": {"notes/rl.md": "high"}}]', RULED_OUT),
    ('[{"query": "x", "relevant": {"notes/rl.md": true}}]', RULED_OUT),
    ('[{"query": "x", "relevant": {"notes/rl.md": -1}}]', RULED_OUT),
    ('[{"relevant": ["notes/rl.md"]}]', RULED_OUT),
    ('[{"query": " ", "relevant": ["notes/rl.md"]}]', RULED_OUT),
    ('[{"query": "x", "relevant": ["notes/rl.md"]}, "a case"]', "case 2 needs"),
    ('{"query": "x"}', "holds no list of cases"),
    ("[]", "holds no list of cases"),
)


@pytest.mark.parametrize("cases, said", REFUSED)
def test_a_file_of_cases_sift_cannot_measure_by_is_refused_saying_which_case(
    home: Path, tmp_path: Path, cases: str, said: str
) -> None:
    (tmp_path / "cases.json").write_text(cases, encoding="utf-8")
    with pytest.raises(InvalidRequestError, match=said) as refused:
        cases_of(home, tmp_path / "cases.json")
    assert refused.value.field == "file"


def test_a_file_that_is_no_json_too_long_or_unreadable_is_refused(home: Path, tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text("[{", encoding="utf-8")
    (tmp_path / "long.json").write_bytes(b" " * (MOST_FILE_BYTES + 1))
    for name, said in (("broken.json", "is no JSON"), ("long.json", "is longer than"), ("none.json", "cannot be read")):
        with pytest.raises(InvalidRequestError, match=said):
            cases_of(home, tmp_path / name)
