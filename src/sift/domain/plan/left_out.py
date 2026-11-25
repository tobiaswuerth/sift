from enum import StrEnum


class WhyLeftOut(StrEnum):
    RULES = "its collection's rules leave it out"
    PASSED_OVER = "a folder on its path is hidden, or one sift passes over"
    LINKED_FOLDER = "a folder on its path is a link, which sift does not follow"
    UNOPENABLE = "a folder on its path has a name Windows cannot open, ending in a dot or a space: rename it"
    OUTSIDE = "it leads outside its collection's folder"
    LOOP = "its links go round in a loop"
    RECORD_NAME = "its name reads as a record's id"
    UNNAMED = "its name is no text"
    NO_SUCH_KIND = "sift takes no such file there"
    NO_FILE = "it is no file to read to an end, such as a pipe or a device"
    FOLDER = "it is a folder: name the files in it, or index its collection"
