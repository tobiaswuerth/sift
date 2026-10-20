"""The accelerator: which device sift computes on, how much of its memory it may use, and when models and vectors
leave the GPU. GPU memory and the clock are fakes; PyTorch itself is real."""

import logging
import sys
import types

import pytest

from sift import accelerator
from sift.accelerator import Accelerator

from .conftest import GiB


@pytest.fixture
def make(gpu_memory, clock):
    """An accelerator for a device, with the fake GPU memory and clock."""
    return lambda device: Accelerator(device, memory=gpu_memory, clock=clock)


def batches_of(memory):
    """A function that runs texts in batches on the fake GPU memory, like a model does."""
    return lambda texts, size: [memory.run(len(texts[i : i + size])) for i in range(0, len(texts), size)]


# --- the device ----------------------------------------------------------------


@pytest.mark.parametrize("cuda, mps, device", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")])
def test_the_device_is_cuda_then_mps_then_cpu(monkeypatch, cuda, mps, device):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert accelerator.detect() == device
    assert Accelerator().device == device


@pytest.mark.parametrize(
    "device, bf16, half",
    [
        ("cuda", True, "bfloat16"),
        ("cuda:1", True, "bfloat16"),
        ("cuda", False, None),
        ("mps", True, None),
        ("cpu", True, None),
    ],
)
def test_half_precision_is_bfloat16_where_the_gpu_has_it(make, monkeypatch, device, bf16, half):
    # EmbeddingGemma overflows in fp16 now and then into NaN vectors; bfloat16 has the range of float32.
    import torch

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: bf16)
    dtype = make(device).half_precision
    assert (None if dtype is None else str(dtype).removeprefix("torch.")) == half


@pytest.mark.parametrize("device, gpu", [("cuda", True), ("cuda:1", True), ("mps", True), ("cpu", False)])
def test_vectors_are_searched_in_fp16_on_gpus_only(make, device, gpu):
    assert (make(device).gpu, make(device).fp16) == (gpu, gpu)  # CPUs are slow at fp16


def test_harmless_torch_windows_note_is_silenced():
    redirects = logging.getLogger("torch.distributed.elastic.multiprocessing.redirects")
    assert redirects.getEffectiveLevel() >= logging.ERROR


# --- the memory budget -----------------------------------------------------------


def test_budget_leaves_room_for_other_programs(make, gpu_memory, capsys):
    cuda = make("cuda")
    cuda.budget()
    cuda.budget()  # set once
    assert gpu_memory.budgets == [(pytest.approx((20 - 2.4) / 24), 0)]  # free minus 10% of the GPU
    assert "using at most 17.6 of 24.0 GB of GPU memory" in capsys.readouterr().err


def test_budget_counts_memory_other_programs_hold(make, gpu_memory):
    # On Windows (WDDM), CUDA reports memory held by other programs as free, since the driver could page it out.
    # NVML, which nvidia-smi uses, does not; the smaller of the two decides.
    gpu_memory.free, gpu_memory.used = 22.5 * GiB, 13.5 * GiB  # as measured on an RTX 4090 under Windows
    make("cuda").budget()
    assert gpu_memory.budgets == [(pytest.approx((24 - 13.5 - 2.4) / 24), 0)]


def test_budget_without_nvml_uses_cuda(make, gpu_memory, monkeypatch):
    def no_nvml(device=None):
        raise ModuleNotFoundError("nvidia-ml-py does not seem to be installed")

    monkeypatch.setattr(gpu_memory, "device_memory_used", no_nvml)
    make("cuda").budget()
    assert gpu_memory.budgets == [(pytest.approx((20 - 2.4) / 24), 0)]


def test_headroom_is_at_least_one_gb(make, gpu_memory):
    gpu_memory.free, gpu_memory.total = 6 * GiB, 8 * GiB
    make("cuda:1").budget()
    assert gpu_memory.budgets == [(pytest.approx(5 / 8), 1)]


def test_too_little_free_memory_means_the_cpu(make, gpu_memory, capsys):
    gpu_memory.free = 2 * GiB  # less than the 2 GiB budget the models need, after headroom
    cuda = make("cuda")
    assert cuda.budget() is None
    assert (cuda.device, cuda.gpu) == ("cpu", False)
    assert gpu_memory.budgets == []
    assert "only 2.0 GB of GPU memory is free; using the CPU" in capsys.readouterr().err


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_no_budget_off_cuda(make, gpu_memory, device):
    assert make(device).budget() is None
    assert gpu_memory.budgets == []


# --- batches: retried smaller when the GPU runs out of memory ----------------------


@pytest.mark.parametrize("gpu", ["cuda", "mps"])
def test_out_of_memory_halves_the_batch_size(make, gpu_memory, monkeypatch, capsys, gpu):
    import torch

    emptied = []
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: emptied.append("mps"))
    monkeypatch.setattr(gpu_memory, "empty_cache", lambda: emptied.append("cuda"))
    oom = (
        torch.cuda.OutOfMemoryError("CUDA out of memory")
        if gpu == "cuda"
        else RuntimeError("MPS backend out of memory")
    )

    def run(texts, size):
        if size > 20:
            raise oom
        return size

    device = make(gpu)
    assert device.run(run, ["a", "b"]) == 16
    assert device.batch_size == 16
    assert emptied == [gpu] * 3  # 128, 64 and 32 failed
    assert "batch size now 16" in capsys.readouterr().err


def test_out_of_memory_at_batch_size_one_raises(make):
    import torch

    def run(texts, size):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    cuda = make("cuda")
    with pytest.raises(RuntimeError, match="out of memory"):
        cuda.run(run, ["a"])
    assert cuda.batch_size == 1


def test_other_runtime_errors_are_not_retried(make):
    def run(texts, size):
        raise RuntimeError("Expected all tensors to be on the same device")

    cuda = make("cuda")
    with pytest.raises(RuntimeError, match="same device"):
        cuda.run(run, ["a"])
    assert cuda.batch_size == 128


# --- calibration: the largest batch that fits the budget ---------------------------


def test_calibration_fills_the_budget(make, gpu_memory, capsys):
    cuda = make("cuda")
    size = cuda.calibrate(gpu_memory.run)
    # budget 17.6 GiB; 90% of it minus 1 GiB of weights leaves 14.84 GiB for 0.1 + 0.05 per text
    assert size == cuda.batch_size == 1 + int((0.9 * 17.6 - 1 - 0.15) / 0.05) == 294
    assert gpu_memory.batches == [1, 2, 4, 8, 16, 32, 64, 128, 256]  # doubled until half the budget was in use
    assert "batch size 294" in capsys.readouterr().err
    cuda.run(batches_of(gpu_memory), ["x"] * 600)  # the calibrated size fits: no out-of-memory, no halving
    assert cuda.batch_size == 294


def test_calibration_stops_at_the_first_out_of_memory(make, gpu_memory):
    gpu_memory.oom_from = 16  # fails earlier than the linear model predicts
    assert make("cuda").calibrate(gpu_memory.run) == 15  # never a size that already failed
    assert gpu_memory.batches == [1, 2, 4, 8, 16]


@pytest.mark.parametrize("per_text", [0.001 * GiB, 0])  # the second: tiny texts may not change the peak at all
def test_calibration_is_capped(make, gpu_memory, per_text):
    gpu_memory.per_text = per_text
    assert make("cuda").calibrate(gpu_memory.run) == accelerator.MAX_BATCH


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_no_calibration_off_cuda(make, gpu_memory, device):
    assert make(device).calibrate(gpu_memory.run) == 128
    assert gpu_memory.batches == []


# --- residency: on the GPU while in use, back in RAM when idle ---------------------


class Thing:
    """Something that moves between RAM and the GPU, like a model or the vectors."""

    def __init__(self, device):
        self.device, self.place = device, "ram"

    def use(self):
        self.device.acquire(id(self), lambda: setattr(self, "place", "gpu"), lambda: setattr(self, "place", "ram"))


def test_things_go_to_the_gpu_when_used_and_back_when_idle(make, gpu_memory, clock, capsys):
    cuda = make("cuda")
    thing = Thing(cuda)
    thing.use()
    assert thing.place == "gpu"
    clock.advance(accelerator.IDLE_SECONDS - 0.5)
    assert not cuda.release_idle()
    clock.advance(0.5)  # exactly IDLE_SECONDS since the last use
    assert cuda.release_idle()
    assert thing.place == "ram"
    assert gpu_memory.emptied  # the memory goes back to the GPU, not just to PyTorch's cache
    assert "moved off the GPU" in capsys.readouterr().err
    clock.advance(10 * accelerator.IDLE_SECONDS)
    assert not cuda.release_idle()  # nothing left to release


def test_every_use_restarts_the_idle_time(make, clock):
    cuda = make("cuda")
    thing = Thing(cuda)
    thing.use()
    clock.advance(accelerator.IDLE_SECONDS - 1)
    thing.use()  # already on the GPU, still counts as use
    clock.advance(accelerator.IDLE_SECONDS - 1)
    assert not cuda.release_idle()
    clock.advance(1)
    assert cuda.release_idle()


def test_idle_release_on_the_limit_despite_rounding(make, clock):
    # (a + 300) - a is 299.9999999999982 for this clock reading: comparing the idle time with the limit missed
    # the release by a rounding error, and failed two CI jobs.
    clock.now = 16165.871537361332
    thing = Thing(make("cuda"))
    thing.use()
    clock.advance(accelerator.IDLE_SECONDS)
    assert thing.device.release_idle()
    assert thing.place == "ram"


def test_nothing_moves_on_the_cpu(make, clock):
    cpu = make("cpu")
    thing = Thing(cpu)
    thing.use()
    assert thing.place == "ram"
    clock.advance(10 * accelerator.IDLE_SECONDS)
    assert not cpu.release_idle()


def test_the_budget_is_refreshed_when_coming_back(make, gpu_memory, clock):
    cuda = make("cuda")
    cuda.budget()
    Thing(cuda).use()
    clock.advance(accelerator.IDLE_SECONDS)
    assert cuda.release_idle()
    gpu_memory.free = 10 * GiB  # another program took more memory meanwhile
    Thing(cuda).use()
    assert gpu_memory.budgets[-1] == (pytest.approx((10 - 2.4) / 24), 0)


def test_coming_back_to_a_busy_gpu_means_the_cpu(make, gpu_memory, clock, capsys):
    # Another program took the GPU memory while sift was idle. Moving back anyway could crash that program.
    cuda = make("cuda")
    cuda.budget()
    Thing(cuda).use()
    clock.advance(accelerator.IDLE_SECONDS)
    assert cuda.release_idle()
    gpu_memory.free = 2 * GiB
    thing = Thing(cuda)
    thing.use()
    assert (cuda.device, thing.place) == ("cpu", "ram")
    assert len(gpu_memory.budgets) == 1  # no budget for memory that isn't there
    assert "using the CPU" in capsys.readouterr().err
    assert cuda.calibrate(gpu_memory.run) == 128  # and no calibration on a GPU given up


# --- an NVIDIA GPU that PyTorch can't use -----------------------------------------


@pytest.fixture
def nvml(monkeypatch):
    """A fake NVML (nvidia-ml-py) that reports one NVIDIA GPU."""
    module = types.ModuleType("pynvml")
    module.gpus = 1
    module.nvmlInit = lambda: None
    module.nvmlDeviceGetCount = lambda: module.gpus
    monkeypatch.setitem(sys.modules, "pynvml", module)
    return module


@pytest.mark.parametrize(
    "cuda_build, hint",
    [
        (None, "uv tool install --reinstall --torch-backend auto sift-rag"),  # uv environments have no pip
        (None, "--index-url https://download.pytorch.org/whl/cu128"),
        ("12.8", "update the NVIDIA driver"),
    ],
)
def test_explains_an_unused_nvidia_gpu(make, nvml, monkeypatch, capsys, cuda_build, hint):
    import torch

    monkeypatch.setattr(torch.version, "cuda", cuda_build)
    cpu = make("cpu")
    cpu.budget()
    cpu.budget()
    err = capsys.readouterr().err
    assert err.count("found an NVIDIA GPU") == 1  # once, not per model
    assert hint in err


def test_no_hint_without_an_nvidia_gpu(make, nvml, capsys):
    nvml.gpus = 0
    make("cpu").budget()
    assert "NVIDIA" not in capsys.readouterr().err


def test_no_hint_without_nvml(make, nvml, monkeypatch, capsys):
    def broken():
        raise OSError("NVML Shared Library Not Found")

    monkeypatch.setattr(nvml, "nvmlInit", broken)
    make("cpu").budget()
    assert "NVIDIA" not in capsys.readouterr().err


def test_no_hint_on_apple_gpus(make, nvml, capsys):
    make("mps").budget()
    assert "NVIDIA" not in capsys.readouterr().err


@pytest.mark.parametrize(
    "platform, command",
    [
        # Windows CUDA wheels bundle their CUDA libraries; Linux ones depend on nvidia-* packages
        (
            "win32",
            "pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu128",
        ),
        ("linux", "pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128"),
    ],
)
def test_cuda_pytorch_command_fits_the_platform(platform, command):
    assert accelerator.cuda_pytorch_command(platform) == command
