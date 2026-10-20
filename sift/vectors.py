"""All chunk vectors of an index as one matrix, to score a query against every chunk at once: in RAM with NumPy, or
on the GPU with PyTorch while in use. Both offer `ids`, `scores(query)` and `close()`."""

import numpy as np


class InRam:
    """The vectors in RAM, in float32 (CPUs are slow at fp16), scored with NumPy: no PyTorch needed."""

    def __init__(self, ids, matrix):
        self.ids, self.matrix = ids, matrix.astype(np.float32)

    def scores(self, query):
        """Cosine similarity of the query vector with every chunk (all vectors are normalised)."""
        return self.matrix @ query.astype(np.float32)

    def close(self):
        """Nothing to give back."""


class OnGpu:
    """The vectors as a tensor that moves to the accelerator's GPU while in use and back to RAM when idle. If sift
    gave up the GPU meanwhile, they are scored in RAM, in float32."""

    def __init__(self, ids, matrix, accelerator):
        import torch

        host = torch.from_numpy(matrix)
        self.ids, self.accelerator = ids, accelerator
        self.host = host if accelerator.fp16 else host.float()
        self.mat = None  # the copy on the GPU, while there
        self._key = ("vectors", id(self))

    def scores(self, query):
        """Cosine similarity of the query vector with every chunk, as a NumPy array."""
        import torch

        self.accelerator.acquire(
            self._key, lambda: setattr(self, "mat", self.host.to(self.accelerator.device)), self._off
        )
        if self.mat is None:  # sift gave up the GPU while the vectors were away
            self.host = self.mat = self.host.float()
        q = torch.from_numpy(query).to(self.mat.device, self.mat.dtype)
        return (self.mat @ q).float().cpu().numpy()

    def _off(self):
        self.mat = None  # the copy in RAM stays

    def close(self):
        """Replaced by newer vectors: the GPU need not move these back any more."""
        self.accelerator.discard(self._key)
