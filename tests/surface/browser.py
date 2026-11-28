class BrowserAskedError(Exception):
    pass


def open_new_tab(url: str) -> bool:
    raise BrowserAskedError(url)
