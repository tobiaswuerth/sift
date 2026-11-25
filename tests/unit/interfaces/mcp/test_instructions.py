from sift.api.collections import CollectionStatus, Counted, Status, Unavailable
from sift.api.errors import Problem
from sift.interfaces.mcp.instructions import HOW, NONE_YET, instructions
from sift.interfaces.mcp.offered import TOOLS

CLIENTS_KEEP = 2_048


def counted(documents: int, fields: tuple[str, ...] = (), links: int = 0, unresolved: int = 0) -> Counted:
    return Counted(documents, documents, 0, 0, (), links, unresolved, fields=fields)


def test_how_to_search_comes_first_and_fits_what_clients_keep_of_instructions() -> None:
    assert len(HOW) <= CLIENTS_KEEP
    assert instructions(Status("/home", ())) == f"{HOW}\n\n{NONE_YET}"


def test_each_collection_is_named_with_its_size_fields_and_links() -> None:
    collections = (
        CollectionStatus("notes", "/data/notes", "/s/notes", counted(1, ("tags", "created"), links=3, unresolved=1)),
        CollectionStatus("papers", "/data/papers", "/s/papers", counted(2_500), active=False),
        CollectionStatus(
            "old", "/data/old", "/s/old", Unavailable(Problem.STORE_DAMAGED, "its index is damaged", "rebuild")
        ),
    )
    assert instructions(Status("/home", collections)).split("\n\n")[-1].splitlines()[1:] == [
        "- notes (/data/notes): 1 document; front matter: tags, created; 3 links, 1 unresolved",
        "- papers (/data/papers): 2,500 documents; inactive",
        "- old (/data/old): its index is damaged",
    ]


def test_every_tool_s_description_fits_what_clients_keep() -> None:
    assert [each.name for each in TOOLS if len(each.description) > CLIENTS_KEEP // 2] == []
