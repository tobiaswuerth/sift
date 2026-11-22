import pytest

from sift.interfaces.console.shell import command


@pytest.mark.parametrize(
    "platform, path, text",
    [
        ("win32", "C:\\My Notes", 'sift add "C:\\My Notes"'),
        ("win32", "C:\\notes", "sift add C:\\notes"),
        ("linux", "/home/me/My Notes", "sift add '/home/me/My Notes'"),
        ("darwin", "/notes", "sift add /notes"),
    ],
)
def test_a_command_to_paste_is_quoted_for_the_shells_there(platform: str, path: str, text: str) -> None:
    assert command(("sift", "add", path), platform) == text
