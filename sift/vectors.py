"""All chunk vectors of an index as one matrix, to score a query against every chunk at once: in RAM with NumPy, or
on the GPU with PyTorch while in use; and `patch`ed, to follow a few chunks changing without loading them all again.
All offer `ids`, in chunk order, `scores(query)`, one per id, and `close()`."""

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


class Patched:
    """Vectors as loaded, `base`, but for those of the chunks since removed (`keep` False, or None for none removed),
    and with those of chunks since embedded, scored in RAM: `added` ids, the rows of `matrix`."""

    def __init__(self, base, keep, added, matrix):
        self.base, self.keep, self.added, self.matrix = base, keep, added, matrix
        self.ids = np.concatenate([base.ids if keep is None else base.ids[keep], added])

    def scores(self, query):
        scores = self.base.scores(query)
        return np.concatenate(
            [scores if self.keep is None else scores[self.keep], self.matrix @ query.astype(np.float32)]
        )

    def close(self):
        self.base.close()


def patch(vectors, removed, ids, matrix):
    """`vectors` without those of the chunks `removed` and with those of the chunks `ids`, the rows of `matrix`: a
    note indexed again while the server runs, without loading millions of vectors again. A chunk of `ids` that the
    vectors hold already gets the new vector: SQLite hands out the ids of deleted chunks again. None if a new chunk
    would come before an old one: the ids must stay in order, and a full load sorts them best."""
    if isinstance(vectors, Patched):
        base, keep, added, rows = vectors.base, vectors.keep, vectors.added, vectors.matrix
    else:
        base, keep, added, rows = vectors, None, np.empty(0, np.int64), np.empty((0, matrix.shape[1]), np.float32)
    removed = np.concatenate([np.asarray(removed, np.int64), np.asarray(ids, np.int64)])
    gone = np.isin(base.ids, removed)
    if gone.any():
        keep = ~gone if keep is None else keep & ~gone
    still = ~np.isin(added, removed)
    added = np.concatenate([added[still], np.asarray(ids, np.int64)])
    rows = np.concatenate([rows[still], matrix.astype(np.float32)])
    order = np.argsort(added, kind="stable")
    old = base.ids if keep is None else base.ids[keep]
    if len(added) and len(old) and added[order[0]] <= old[-1]:
        return None
    return Patched(base, keep, added[order], rows[order])
