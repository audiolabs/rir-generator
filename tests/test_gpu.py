"""Tests for the PyTorch GPU backend (rir_generator.gpu).

Skipped automatically when torch is not installed.

The key assertion in every comparison test is::

    np.allclose(h_cpp, h_gpu, atol=1e-6)

In practice the maximum observed difference is on the order of 1e-15
(floating-point rounding), so atol=1e-6 is a very conservative bound.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # skip entire module if torch absent

import rir_generator
from rir_generator import gpu as rir_gpu

# ---------------------------------------------------------------------------
# Shared fixtures (representative subset of test_example.py parameters)
# ---------------------------------------------------------------------------

@pytest.fixture(params=[16000, 44100.5])
def fs(request):
    return request.param


@pytest.fixture(
    params=[
        (1, [2, 1.5, 2]),
        (2, [[2, 1.5, 2], [1, 1.5, 2]]),
    ]
)
def nMics_r(request):
    return request.param


@pytest.fixture
def nMics(nMics_r):
    return nMics_r[0]


@pytest.fixture
def r(nMics_r):
    return nMics_r[1]


@pytest.fixture(params=[[2, 3.5, 2]])
def s(request):
    return request.param


@pytest.fixture(params=[[5, 4, 6]])
def L(request):
    return request.param


@pytest.fixture(params=[340])
def c(request):
    return request.param


@pytest.fixture(params=[2048])
def n(request):
    return request.param


@pytest.fixture(
    params=[
        (0.4, None),
        (None, [0.1, 0.1, 0.2, 0.2, 0.3, 0.3]),
    ]
)
def reverberation_time_beta(request):
    return request.param


@pytest.fixture
def reverberation_time(reverberation_time_beta):
    return reverberation_time_beta[0]


@pytest.fixture
def beta(reverberation_time_beta):
    return reverberation_time_beta[1]


@pytest.fixture(
    params=[
        rir_generator.mtype.omnidirectional,
        rir_generator.mtype.hypercardioid,
    ]
)
def mtype(request):
    return request.param


@pytest.fixture(params=[2, -1])
def order(request):
    return request.param


@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


@pytest.fixture(params=[0, [np.pi / 2, 0]])
def orientation(request):
    return request.param


@pytest.fixture(params=[True, False])
def hp_filter(request):
    return request.param


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _call_both(c, fs, r, s, L, reverberation_time, beta, n,
               mtype, order, dim, orientation, hp_filter):
    """Return (h_cpp, h_gpu) for identical parameters."""
    kwargs = dict(
        c=c, fs=fs, r=r, s=s, L=L,
        reverberation_time=reverberation_time,
        beta=beta,
        nsample=n,
        mtype=mtype,
        order=order,
        dim=dim,
        orientation=orientation,
        hp_filter=hp_filter,
    )
    h_cpp = rir_generator.generate(**kwargs)
    h_gpu = rir_gpu.generate(**kwargs, device="cpu")
    return h_cpp, h_gpu


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_gpu_output_shape(c, fs, r, nMics, s, L, n,
                          reverberation_time, beta, mtype,
                          order, dim, orientation, hp_filter):
    """GPU backend returns the correct array shape."""
    h = rir_gpu.generate(
        c=c, fs=fs, r=r, s=s, L=L,
        reverberation_time=reverberation_time,
        beta=beta,
        nsample=n,
        mtype=mtype,
        order=order,
        dim=dim,
        orientation=orientation,
        hp_filter=hp_filter,
        device="cpu",
    )
    assert h.shape == (n, nMics)
    assert not np.all(np.isclose(h, 0))


def test_gpu_matches_cpp(c, fs, r, s, L, n,
                         reverberation_time, beta, mtype,
                         order, dim, orientation, hp_filter):
    """GPU backend output is numerically identical to the C++ reference."""
    h_cpp, h_gpu = _call_both(
        c, fs, r, s, L, reverberation_time, beta, n,
        mtype, order, dim, orientation, hp_filter,
    )
    assert h_cpp.shape == h_gpu.shape
    assert np.allclose(h_cpp, h_gpu, atol=1e-6), (
        f"Max |Δ| = {np.max(np.abs(h_cpp - h_gpu)):.3e}"
    )


def test_gpu_multiple_mics_consistent(
    c, fs, s, L, n, reverberation_time, beta,
    mtype, order, dim, orientation, hp_filter,
):
    """Single-mic and multi-mic GPU calls agree on the shared microphone."""
    kwargs = dict(
        c=c, fs=fs, s=s, L=L,
        reverberation_time=reverberation_time,
        beta=beta,
        nsample=n,
        mtype=mtype,
        order=order,
        dim=dim,
        orientation=orientation,
        hp_filter=hp_filter,
        device="cpu",
    )
    h1 = rir_gpu.generate(r=[2, 1.5, 2], **kwargs)
    h2 = rir_gpu.generate(r=[[2, 1.5, 2], [1, 1.5, 2]], **kwargs)
    assert np.allclose(h1[:, 0], h2[:, 0])


@pytest.mark.parametrize(
    "r, s",
    [
        ([2, 3.5, 2], [2, 5.5, 2]),
        ([2, 5.5, 2], [2, 1.5, 2]),
    ],
)
def test_gpu_outside_room(r, s):
    """GPU backend raises ValueError when positions are outside the room."""
    with pytest.raises(ValueError):
        rir_gpu.generate(
            c=340, fs=16000, r=r, s=s, L=[5, 4, 6],
            reverberation_time=0.4,
            device="cpu",
        )


@pytest.mark.parametrize(
    "beta",
    [
        [(0.1, 0.1, 0.2), (0.2, 0.3, 0.3)],
    ],
)
def test_gpu_beta_shape(beta):
    """GPU backend raises AssertionError for invalid beta shape."""
    with pytest.raises(AssertionError):
        rir_gpu.generate(
            c=340, fs=16000, r=[2, 1.5, 2], s=[2, 3.5, 2],
            L=[5, 4, 6], beta=beta,
            device="cpu",
        )
