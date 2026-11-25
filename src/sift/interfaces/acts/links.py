from sift.api.links import Links, ListUnresolved, ShowLinks, Unresolved
from sift.app import links
from sift.interfaces.acts.acting import Act, Acting


def _links(acting: Acting, request: ShowLinks) -> Links:
    return links.links(acting.settings.home, request)


def _unresolved(acting: Acting, request: ListUnresolved) -> Unresolved:
    return links.unresolved(acting.settings.home, request)


LINKS = Act("links", ShowLinks, Links, _links)
UNRESOLVED = Act("unresolved", ListUnresolved, Unresolved, _unresolved)
