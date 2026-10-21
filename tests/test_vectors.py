"""All chunk vectors as one matrix: in RAM with NumPy, or on the GPU with PyTorch while in use."""

import numpy as np
import pytest

from sift import accelerator
from sift.accelerator import Accelerator
from sift.vectors import InRam, OnGpu, patch

from .conftest import GiB, PretendGpu

IDS = np.array([10, 20, 30])
MATRIX = np.array([[1, 0], [0, 1], [0.6, 0.8]], np.float16)


@pytest.fixture
def pretend(gpu_memory, clock):
    return PretendGpu("cpu", memory=gpu_memory, clock=clock)


@pytest.mark.parametrize("make", [lambda acc: InRam(IDS, MATRIX), lambda acc: OnGpu(IDS, MATRIX, acc)])
def test_scores_are_dot_products_with_every_vector(pretend, make):
    vectors = make(pretend)
    assert vectors.ids.tolist() == [10, 20, 30]
    scores = vectors.scores(np.array([1, 0], np.float16))
    assert isinstance(scores, np.ndarray)
    assert scores.tolist() == pytest.approx([1, 0, 0.6], abs=1e-3)


def test_in_ram_the_vectors_are_float32(pretend):
    assert InRam(IDS, MATRIX).matrix.dtype == np.float32  # CPUs are slow at fp16


def test_on_a_gpu_the_vectors_stay_fp16(pretend):
    import torch

    vectors = OnGpu(IDS, MATRIX, pretend)
    vectors.scores(np.array([1, 0], np.float16))
    assert vectors.host.dtype == vectors.mat.dtype == torch.float16


def test_the_vectors_leave_the_gpu_when_idle(pretend, clock):
    vectors = OnGpu(IDS, MATRIX, pretend)
    first = vectors.scores(np.array([1, 0], np.float16)).tolist()
    assert vectors.mat is not None
    clock.advance(accelerator.IDLE_SECONDS)
    assert pretend.release_idle()
    assert vectors.mat is None  # only the copy in RAM is left
    assert vectors.scores(np.array([1, 0], np.float16)).tolist() == first
    assert vectors.mat is not None


def test_closed_vectors_are_forgotten_by_the_gpu(pretend, clock):
    vectors = OnGpu(IDS, MATRIX, pretend)
    vectors.scores(np.array([1, 0], np.float16))
    vectors.close()  # replaced by newer ones
    clock.advance(accelerator.IDLE_SECONDS)
    assert not pretend.release_idle()  # nothing of theirs left to move


def test_vectors_stay_in_ram_in_float32_when_the_gpu_is_gone(gpu_memory, clock):
    import torch

    cuda = Accelerator("cuda", memory=gpu_memory, clock=clock)
    vectors = OnGpu(IDS, MATRIX, cuda)
    cuda.budget()
    cuda.acquire("models", lambda: None, lambda: None)
    clock.advance(accelerator.IDLE_SECONDS)
    assert cuda.release_idle()
    gpu_memory.free = 2 * GiB  # meanwhile another program took the GPU
    assert vectors.scores(np.array([1, 0], np.float16)).tolist() == pytest.approx([1, 0, 0.6], abs=1e-3)
    assert vectors.mat.dtype == torch.float32  # searched in RAM, where fp16 is slow
    assert cuda.device == "cpu"


# --- vectors patched instead of loaded anew ---------------------------------


def scored(vectors):
    scores = vectors.scores(np.array([1, 0], np.float16))
    return {chunk: round(float(score), 3) for chunk, score in zip(vectors.ids.tolist(), scores)}


def rows(*vectors):
    return np.array(vectors, np.float16).reshape(-1, 2)


@pytest.mark.parametrize("make", [lambda acc: InRam(IDS, MATRIX), lambda acc: OnGpu(IDS, MATRIX, acc)])
def test_patched_vectors_score_as_if_loaded_anew(pretend, make):
    # A note indexed again while the server runs: its old vectors go and its new ones come, without loading the
    # millions of others again.
    vectors = patch(make(pretend), removed=[20], ids=[40], matrix=rows([0.8, 0.6]))
    assert scored(vectors) == {10: 1.0, 30: 0.6, 40: 0.8}
    vectors = patch(vectors, removed=[40, 10, 99], ids=[41, 42], matrix=rows([0, 1], [1, 0]))
    assert scored(vectors) == {30: 0.6, 41: 0.0, 42: 1.0}
    vectors = patch(vectors, removed=[41], ids=[], matrix=rows())
    assert scored(vectors) == {30: 0.6, 42: 1.0}
    assert patch(vectors, removed=[], ids=[], matrix=rows()).ids.tolist() == [30, 42]


def test_patched_vectors_keep_chunk_order():
    # Searches find a chunk's score by binary search over the ids: a new chunk among the old ones would need sorting
    # them all, which a full load does better.
    assert patch(InRam(IDS, MATRIX), removed=[], ids=[25], matrix=rows([1, 0])) is None
    assert patch(InRam(IDS, MATRIX), removed=[30], ids=[30], matrix=rows([1, 0])).ids.tolist() == [10, 20, 30]


def test_closing_patched_vectors_closes_those_beneath(pretend, clock):
    vectors = patch(OnGpu(IDS, MATRIX, pretend), removed=[10], ids=[], matrix=rows())
    vectors.scores(np.array([1, 0], np.float16))
    vectors.close()
    clock.advance(accelerator.IDLE_SECONDS)
    assert not pretend.release_idle()
