import logging
import warnings

import pytest

from sift.inference import local, ocr
from sift.inference.loading import REDIRECTS, LocalLoader, OcrModule, ocr_module


def test_the_note_pytorch_logs_on_a_mac_as_the_models_load_is_silenced() -> None:
    ocr_module()
    assert logging.getLogger(REDIRECTS).getEffectiveLevel() == logging.ERROR


def test_pytorch_s_warning_of_a_virtual_mac_s_gpu_goes_unheard_once_the_models_load_and_others_are_heard() -> None:
    ocr_module()
    warnings.warn("Skipping device Apple Paravirtual device that does not support Metal 2.0", UserWarning, stacklevel=1)
    with pytest.warns(UserWarning, match="MPS is deprecated"):
        warnings.warn("MPS is deprecated", UserWarning, stacklevel=1)


def test_the_modules_loaded_late_are_what_loading_takes_them_for() -> None:
    loader: LocalLoader = local
    reader: OcrModule = ocr
    assert (loader.local, ocr_module()) == (local.local, reader)
