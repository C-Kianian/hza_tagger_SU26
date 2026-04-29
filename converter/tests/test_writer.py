"""Unit tests for the H5Writer."""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from common.io import JET_DTYPE, TRACK_DTYPE, LABEL_DTYPE, JETS_DATASET, TRACKS_DATASET, LABELS_DATASET
from common.variables import N_TRACKS
from converter.processors.writer import H5Writer


def _make_chunk(n: int):
    jets   = np.zeros(n, dtype=JET_DTYPE)
    jets["pt"]   = np.random.uniform(30, 500, n)
    jets["eta"]  = np.random.uniform(-2.4, 2.4, n)
    jets["phi"]  = np.random.uniform(-np.pi, np.pi, n)
    jets["mass"] = np.random.uniform(0, 50, n)

    tracks = np.zeros((n, N_TRACKS), dtype=TRACK_DTYPE)
    tracks["valid"] = True

    labels = np.zeros(n, dtype=LABEL_DTYPE)
    labels["a_jet"] = np.random.randint(0, 2, n)
    return jets, tracks, labels


def test_single_chunk():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "out.h5"
        with H5Writer(path) as w:
            j, t, l = _make_chunk(100)
            w.write_chunk(j, t, l)
            w.finalize()
        with h5py.File(path) as f:
            assert f[JETS_DATASET].shape == (100,)
            assert f[TRACKS_DATASET].shape == (100, N_TRACKS)
            assert f[LABELS_DATASET].shape == (100,)


def test_multiple_chunks_grow():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "out.h5"
        with H5Writer(path) as w:
            for _ in range(3):
                j, t, l = _make_chunk(50)
                w.write_chunk(j, t, l)
            w.finalize()
        with h5py.File(path) as f:
            assert f[JETS_DATASET].shape[0] == 150
            assert f[TRACKS_DATASET].shape[0] == 150


def test_empty_chunk_skipped():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "out.h5"
        with H5Writer(path) as w:
            j, t, l = _make_chunk(0)
            w.write_chunk(j, t, l)
            j, t, l = _make_chunk(10)
            w.write_chunk(j, t, l)
            w.finalize()
        with h5py.File(path) as f:
            assert f[JETS_DATASET].shape[0] == 10


def test_pt_values_preserved():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "out.h5"
        j, t, l = _make_chunk(20)
        with H5Writer(path) as w:
            w.write_chunk(j, t, l)
            w.finalize()
        with h5py.File(path) as f:
            np.testing.assert_allclose(f[JETS_DATASET]["pt"][:], j["pt"])
