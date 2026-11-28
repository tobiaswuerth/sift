from sift.inference.held import OnDevice
from sift.inference.local import Local


def placed(models: Local) -> OnDevice:
    assert isinstance(models.state, OnDevice)
    return models.state
