import logging

from sift.inference.loading import REDIRECTS, ocr_module


def test_the_note_pytorch_logs_on_a_mac_as_the_models_load_is_silenced() -> None:
    ocr_module()
    assert logging.getLogger(REDIRECTS).getEffectiveLevel() == logging.ERROR
