from sift.interfaces.web.view import PAGE, PageView, filtered, filters_of, folder_glob, href, view_of


def test_an_address_is_read_into_a_view_and_odd_values_fall_back() -> None:
    pairs = [("q", " dark matter "), ("mode", "fuzzy"), ("path", "/2024//x/"), ("f", "year=2024"), ("f", "bad")]
    pairs += [("k", "99999999999"), ("at", "-3"), ("doc", "notes/a.md")]
    assert view_of(pairs) == PageView(q="dark matter", path="2024/x/", f=("year=2024",), doc="notes/a.md")


def test_odd_page_sizes_show_a_page_and_large_ones_are_capped() -> None:
    assert [view_of([("k", given)]).k for given in ("0", "abc", "7", "100000")] == [PAGE, PAGE, 7, 500]


def test_an_address_holds_what_differs_from_the_default_alone_and_reads_back_alike() -> None:
    view = PageView(q="a&b", f=("tag=x", "tag=y"), doc="n/ü.md")
    written = href(view)
    assert written == "/ui?q=a%26b&f=tag%3Dx&f=tag%3Dy&doc=n%2F%C3%BC.md"
    assert href(PageView()) == "/ui"


def test_a_filter_is_added_or_taken_away_on_the_first_page() -> None:
    view = PageView(q="x", f=("tag=a",), k=90)
    assert filtered(view, "tag", "b") == "/ui?q=x&f=tag%3Da&f=tag%3Db"
    assert filtered(view, "tag", "a") == "/ui?q=x"


def test_filters_of_one_field_are_any_of_their_values() -> None:
    assert filters_of(PageView(f=("tag=a", "year=2024", "tag=b"))) == [
        {"field": "tag", "operator": "=", "values": ["a", "b"]},
        {"field": "year", "operator": "=", "values": ["2024"]},
    ]


def test_a_folder_s_glob_takes_its_brackets_stars_and_question_marks_as_written() -> None:
    assert folder_glob("z/[x]*?/") == "z/[[]x][*][?]/*"
