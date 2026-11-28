NVML_ERROR_NOT_SUPPORTED: int

class NVMLError(Exception):
    def __init__(self, value: int) -> None: ...
