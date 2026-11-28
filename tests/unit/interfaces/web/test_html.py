from sift.interfaces.web.html import Html, joined, tag


def test_text_is_escaped_and_built_html_is_not() -> None:
    built = tag("p", {"class": "hit"}, 'a < b & "c"', tag("mark", {}, "<b>"), 3)
    assert built == Html('<p class="hit">a &lt; b &amp; &quot;c&quot;<mark>&lt;b&gt;</mark>3</p>')


def test_attributes_are_escaped_and_left_out_when_absent_or_bare_when_true() -> None:
    built = tag("a", {"href": '/ui?q="x"&y', "title": None, "hidden": False, "data-on": True})
    assert built == Html('<a href="/ui?q=&quot;x&quot;&amp;y" data-on></a>')


def test_void_elements_have_no_end_and_parts_join_in_order() -> None:
    assert tag("input", {"name": "q"}) == Html('<input name="q">')
    assert joined(["a", tag("br"), ("b", "<")]) == Html("a<br>b&lt;")
