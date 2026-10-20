"""The device sift computes on, and the share of its memory sift may use: an NVIDIA GPU, an Apple GPU or the CPU.

Models and vectors wait in RAM and move to the GPU while in use (`acquire`); after IDLE_SECONDS without use they move
back (`release_idle`), so the GPU stays free for other programs. On NVIDIA GPUs sift caps itself at the memory other
programs leave free, minus headroom, and with too little, it uses the CPU instead.
"""

import contextlib
import logging
import sys
from time import monotonic

import torch

# Loading Sentence Transformers makes PyTorch log a harmless note about distributed training on Windows and macOS.
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

GiB = 2**30
HEADROOM = 0.1  # share of the GPU left free for other programs, at least 1 GiB
MIN_BUDGET = 2 * GiB  # with less GPU memory than this, the models run on the CPU
FILL = 0.9  # share of the memory budget a calibrated batch may use
BATCH_SIZE = 128  # until calibrated on CUDA; a good size for CPUs and Apple GPUs
MAX_BATCH = 1024
IDLE_SECONDS = 300  # unused this long, models and vectors give their GPU memory back
_UNSET = object()


def detect():
    """An NVIDIA GPU (CUDA) if there is one, else an Apple GPU (MPS), else the CPU."""
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def cuda_pytorch_command(platform=sys.platform):
    """How to replace a CPU-only PyTorch with the CUDA build. Windows wheels bundle their CUDA libraries, so
    torch's other dependencies can stay; Linux wheels depend on nvidia-* packages, which must come along."""
    no_deps = "--no-deps " if platform == "win32" else ""
    return (
        f"pip install --force-reinstall {no_deps}torch torchvision --index-url https://download.pytorch.org/whl/cu128"
    )


def _unused_nvidia_gpu():
    """Why an NVIDIA GPU sits unused, if there is one; None otherwise."""
    try:
        import pynvml

        pynvml.nvmlInit()
        if not pynvml.nvmlDeviceGetCount():
            return None
    except Exception:  # no NVML, no driver, no GPU
        return None
    if torch.version.cuda is None:
        return (
            "found an NVIDIA GPU, but this PyTorch build has no CUDA support, so sift runs on the CPU (much slower). "
            "To use the GPU, reinstall with uv: `uv tool install --reinstall --torch-backend auto sift-rag` "
            f"(or run `uvx --torch-backend auto sift-rag ...`), or with pip: `{cuda_pytorch_command()}`"
        )
    return f"found an NVIDIA GPU, but PyTorch (CUDA {torch.version.cuda}) can't use it; update the NVIDIA driver"


class Accelerator:
    """A device and the bookkeeping around it: the memory budget, the batch size, what is on the GPU, when it was
    last used.

    `memory` is CUDA's memory API (torch.cuda) and `clock` a monotonic clock; tests pass fakes of both.
    """

    def __init__(self, device=None, memory=torch.cuda, clock=monotonic):
        self.device = device or detect()
        self.memory, self.clock = memory, clock
        self.batch_size = BATCH_SIZE
        self._budget = _UNSET  # asked once, on first use
        self._resident = {}  # what is on the GPU now: key -> function that moves it back to RAM
        self._released = False  # whether everything left the GPU at some point, so the budget needs a fresh look
        self._last_use = 0.0

    @property
    def kind(self):
        """ "cuda", "mps" or "cpu"."""
        return torch.device(self.device).type

    @property
    def gpu(self):
        """Whether sift computes on a GPU (NVIDIA or Apple)."""
        return self.kind in ("cuda", "mps")

    @property
    def fp16(self):
        """Whether vectors are searched in fp16: on GPUs; CPUs are slow at fp16 matmul."""
        return self.gpu

    @property
    def half_precision(self):
        """The 16-bit type models compute in: bfloat16 on NVIDIA GPUs that have it, None (float32) elsewhere. Never
        fp16: its range is too small for EmbeddingGemma, which overflows in it now and then into NaN vectors;
        bfloat16 has the range of float32."""
        return torch.bfloat16 if self.kind == "cuda" and torch.cuda.is_bf16_supported() else None

    # --- memory ------------------------------------------------------------------

    def budget(self):
        """The GPU memory sift may use, set once as a cap on PyTorch: what other programs leave free now, minus
        headroom. None off CUDA, and when too little is free, which moves sift to the CPU."""
        if self._budget is _UNSET:
            if self.kind != "cuda":
                if self.device == "cpu" and (reason := _unused_nvidia_gpu()):
                    print(reason, file=sys.stderr)
                self._budget = None
            else:
                self._budget = self._set_budget()
        return self._budget

    def _free(self):
        """Free and total memory of the GPU. On Windows (WDDM), CUDA counts memory other programs hold as free,
        since the driver could page it out; NVML, which nvidia-smi reads, does not. The smaller value wins."""
        free, total = self.memory.mem_get_info(self.device)
        with contextlib.suppress(Exception):  # NVML missing or unusable: CUDA's figure is all there is
            free = min(free, total - self.memory.device_memory_used(self.device))
        return free, total

    def _set_budget(self):
        """Cap PyTorch's GPU memory at what other programs leave free, minus headroom. None, and the CPU from now on,
        if too little is free."""
        free, total = self._free()
        budget = free - max(HEADROOM * total, GiB)
        if budget < MIN_BUDGET:
            print(f"only {free / GiB:.1f} GB of GPU memory is free; using the CPU (slower)", file=sys.stderr)
            self.device = "cpu"
            return None
        index = torch.device(self.device).index  # PyTorch wants an index here; plain "cuda" is the current device
        self.memory.set_per_process_memory_fraction(
            budget / total, self.memory.current_device() if index is None else index
        )
        print(f"using at most {budget / GiB:.1f} of {total / GiB:.1f} GB of GPU memory", file=sys.stderr)
        return budget

    def _empty_cache(self):
        """Hand cached memory back to the GPU, not just to PyTorch's cache."""
        if self.kind == "cuda":
            self.memory.empty_cache()
        elif self.kind == "mps":
            torch.mps.empty_cache()

    # --- batches ---------------------------------------------------------------------

    def run(self, fn, items):
        """fn(items, batch_size), halving the batch size whenever the GPU runs out of memory."""
        while True:
            try:
                return fn(items, self.batch_size)
            except RuntimeError as e:  # CUDA raises OutOfMemoryError, a RuntimeError; MPS a plain RuntimeError
                out_of_memory = isinstance(e, torch.cuda.OutOfMemoryError) or "MPS backend out of memory" in str(e)
                if self.batch_size == 1 or not out_of_memory:
                    raise
                self._empty_cache()
                self.batch_size //= 2
                print(f"\nout of GPU memory, batch size now {self.batch_size}", file=sys.stderr)

    def calibrate(self, run):
        """Measure the largest batch that fits the memory budget, and use it from now on. run(n) runs a batch of n
        of the longest texts to come.

        Batches of 1, 2, 4, ... run until one uses half of the budget. Memory grows linearly with the batch size,
        so the last measurement predicts the size that fills FILL of the budget. A size that ran out of memory is
        never chosen. Off CUDA, and on a GPU given up, the batch size stays as it is.
        """
        budget = self.budget()
        if budget is None or self.kind != "cuda":
            return self.batch_size
        base = self.memory.memory_allocated(self.device)
        target = FILL * budget - base

        def cost(n):
            self.memory.reset_peak_memory_stats(self.device)
            run(n)
            return self.memory.max_memory_allocated(self.device) - base

        one = used = cost(1)
        n, failed = 1, MAX_BATCH + 1
        while n < MAX_BATCH and used < target / 2:
            try:
                used = cost(2 * n)
            except torch.cuda.OutOfMemoryError:
                failed = 2 * n
                break
            n *= 2
        self._empty_cache()
        per_text = max((used - one) / (n - 1) if n > 1 else one, 1)  # tiny texts may not change the peak at all
        self.batch_size = max(1, min(failed - 1, MAX_BATCH, 1 + int((target - one) / per_text)))
        print(f"batch size {self.batch_size}, calibrated to the GPU memory budget", file=sys.stderr)
        return self.batch_size

    # --- what is on the GPU ------------------------------------------------------------

    def acquire(self, key, to_gpu, off_gpu):
        """Make sure something is on the GPU: to_gpu() moves it there unless it is already, off_gpu() moves it back.
        Every call counts as use. On the CPU, nothing moves."""
        self._last_use = self.clock()
        self.resume()
        if not self.gpu or key in self._resident:
            return
        to_gpu()
        self._resident[key] = off_gpu

    def discard(self, key):
        """Forget something that left the GPU on its own, such as vectors replaced by newer ones."""
        self._resident.pop(key, None)

    def release_idle(self):
        """Move everything back to RAM once the GPU has been unused for IDLE_SECONDS. Returns whether it did."""
        if not self._resident or self.clock() < self._last_use + IDLE_SECONDS:  # not now - last use: rounding
            return False
        for off_gpu in self._resident.values():
            off_gpu()
        self._resident.clear()
        self._released = True
        self._empty_cache()
        print(f"idle for {IDLE_SECONDS // 60} min: models and vectors moved off the GPU", file=sys.stderr)
        return True

    def resume(self):
        """After an idle release, look at the GPU memory again: other programs may hold more or less of it by now.
        With too little left, sift uses the CPU from then on, as it does when there is too little at start-up."""
        if self._released and self.kind == "cuda":
            self._released = False
            self._budget = self._set_budget()
