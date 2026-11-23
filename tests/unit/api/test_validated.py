from dataclasses import dataclass
from typing import Annotated

import pytest

from sift.api.described import Bounds, Help
from sift.api.errors import InvalidRequestError
from sift.api.validated import validated


@dataclass(frozen=True, slots=True)
class Asked:
    k: Annotated[int, Help("how many"), Bounds(1, 9)] = 5
    name: Annotated[str, Help("a name")] = "x"
    plain: int = 0


@pytest.mark.parametrize("k, said", [(0, "0 is outside 1 to 9"), (10, "10 is outside 1 to 9")])
def test_a_field_outside_its_bounds_is_refused_by_name(k: int, said: str) -> None:
    with pytest.raises(InvalidRequestError, match=said) as refused:
        validated(Asked(k=k))
    assert (refused.value.field, refused.value.remedy) == ("k", "give 1 to 9")


def test_fields_within_their_bounds_pass_and_only_a_request_is_validated() -> None:
    validated(Asked(k=1))
    validated(Asked(k=9))
    with pytest.raises(TypeError, match="is no request"):
        validated(Asked)
