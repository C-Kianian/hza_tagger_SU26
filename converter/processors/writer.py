"""Chunked H5 writer that appends jet / track / label arrays across coffea chunks."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from common.io import JET_DTYPE, LABEL_DTYPE, TRACK_DTYPE, JETS_DATASET, TRACKS_DATASET, LABELS_DATASET
from common.variables import N_TRACKS


class H5Writer:
    """Incrementally writes structured arrays to an HDF5 file.

    Call :meth:`write_chunk` for each coffea chunk.  The datasets are created
    on the first call (resizable along axis-0) and grown on subsequent calls.
    Call :meth:`close` (or use as context manager) when done.
    """

    def __init__(self, path: str | Path, compression: str = "gzip", compression_opts: int = 4):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._f = h5py.File(self._path, "w")
        self._compression = compression
        self._compression_opts = compression_opts
        self._current_size = 0
        self._initialized = False

    # ── context manager ──────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self._f.close()

    # ── public API ───────────────────────────────────────────────────────────

    def write_chunk(
        self,
        jets: np.ndarray,       # shape (N,)    dtype=JET_DTYPE
        tracks: np.ndarray,     # shape (N, T)  dtype=TRACK_DTYPE
        labels: np.ndarray,     # shape (N,)    dtype=LABEL_DTYPE
    ):
        """Append one chunk of jets/tracks/labels to the HDF5 file."""
        n = len(jets)
        if n == 0:
            return

        if not self._initialized:
            self._create_datasets(n, tracks.shape[1])
            self._initialized = True
        else:
            self._extend(n)

        idx = self._current_size - n
        self._f[JETS_DATASET][idx:] = jets
        self._f[TRACKS_DATASET][idx:] = tracks
        self._f[LABELS_DATASET][idx:] = labels

    def finalize(self):
        """Write metadata and flush."""
        self._f.attrs["n_jets"] = self._current_size
        self._f.attrs["n_tracks"] = N_TRACKS
        self._f.flush()

    # ── internals ────────────────────────────────────────────────────────────

    def _create_datasets(self, n: int, n_tracks: int):
        kw = dict(compression=self._compression, compression_opts=self._compression_opts)
        self._f.create_dataset(
            JETS_DATASET,
            shape=(n,),
            maxshape=(None,),
            dtype=JET_DTYPE,
            **kw,
        )
        self._f.create_dataset(
            TRACKS_DATASET,
            shape=(n, n_tracks),
            maxshape=(None, n_tracks),
            dtype=TRACK_DTYPE,
            **kw,
        )
        self._f.create_dataset(
            LABELS_DATASET,
            shape=(n,),
            maxshape=(None,),
            dtype=LABEL_DTYPE,
            **kw,
        )
        self._current_size = n

    def _extend(self, n: int):
        new_size = self._current_size + n
        self._f[JETS_DATASET].resize(new_size, axis=0)
        self._f[TRACKS_DATASET].resize(new_size, axis=0)
        self._f[LABELS_DATASET].resize(new_size, axis=0)
        self._current_size = new_size
