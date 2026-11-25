import json
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from sift.interfaces.cli.arguments import Argument, How, arguments
from sift.interfaces.cli.entries import ENTRIES
from tests.surface import process

type Done = subprocess.CompletedProcess[str]
type Answer = dict[str, object]
type Argv = tuple[str, ...]

FLAGGED = (How.OPTION, How.SWITCH, How.TOGGLE, How.READ)
LISTED = (How.ANY, How.SOME)
SHOWN = 2000


@dataclass
class Smoke:
    home: Path
    model: str = ""
    reranker: str = ""
    exercised: list[Argv] = field(default_factory=list[Argv])

    def ran(self, *args: str, cwd: Path | None = None) -> Done:
        return process.sift(self.home, *args, model=self.model, reranker=self.reranker, cwd=cwd)

    def checked(self, *args: str, holds: Callable[[Done], bool], cwd: Path | None = None) -> Done:
        done = self.ran(*args, cwd=cwd)
        assert holds(done), (args, done.returncode, done.stdout[-SHOWN:], done.stderr[-SHOWN:])
        self.exercised.append(args)
        return done


def answer(done: Done) -> Answer:
    return cast("Answer", json.loads(done.stdout))


def listed(done: Done, key: str) -> list[Answer]:
    return cast("list[Answer]", answer(done)[key])


def hits_of(done: Done) -> list[Answer]:
    return listed(done, "hits")


def ids(done: Done) -> list[str]:
    return [str(hit["id"]) for hit in hits_of(done)]


def finding(*expected: str) -> Callable[[Done], bool]:
    return lambda done: ids(done) == list(expected)


def answering(key: str, value: object) -> Callable[[Done], bool]:
    return lambda done: answer(done)[key] == value


def _forms(argument: Argument) -> frozenset[str]:
    return frozenset(name for name in (argument.flag.long, argument.flag.short) if name is not None)


def _given(forms: frozenset[str], argvs: list[Argv]) -> bool:
    return any(word.split("=", 1)[0] in forms for argv in argvs for word in argv)


def unexercised(ran: list[Argv], names: Iterable[str]) -> list[str]:
    missing: list[str] = []
    for name in names:
        entry = ENTRIES[name]
        argvs = [argv for argv in ran if argv[:1] == (name,)]
        for argument in arguments(entry.request):
            if argument.how in FLAGGED and not _given(_forms(argument), argvs):
                missing.append(f"{name} {min(_forms(argument), key=len)}")
            if argument.how in LISTED and not any(len(argv) > 1 and not argv[1].startswith("-") for argv in argvs):
                missing.append(f"{name} {argument.name.upper()}")
        if not _given(frozenset({"--json"}), argvs):
            missing.append(f"{name} --json")
    return missing


def coloured(ran: list[Argv]) -> bool:
    return _given(frozenset({"--color"}), ran)


def ok(done: Done) -> bool:
    return (done.returncode, done.stderr) == (0, "")


def indexed(smoke: Smoke, notes: Path) -> None:
    smoke.checked(
        "add",
        str(notes),
        "--name",
        "notes",
        holds=lambda done: (
            (done.returncode, done.stdout, done.stderr.split("\n")[0]) == (0, "", f"added  notes  {notes.resolve()}")
        ),
    )
    smoke.checked(
        "index", holds=lambda done: (done.returncode, done.stdout) == (0, "") and "  done  +3  ~0  -0  " in done.stderr
    )
