"""GPU-accelerated Room Impulse Response generator using PyTorch.

Implements the same Image Source Method as the C++ backend (Allen & Berkley 1979,
Peterson 1986), but vectorises *all* image sources simultaneously as PyTorch
tensors.  When a CUDA device is available the computation runs on GPU; otherwise
it falls back to CPU tensors, which are still significantly faster than the
sequential C++ loop thanks to SIMD vectorisation.

The public ``generate()`` function is a drop-in replacement for
``rir_generator.generate()`` and accepts two additional keyword arguments:

* ``device``     – PyTorch device string or object (default: CUDA if available)
* ``chunk_size`` – number of image sources per batch (trade memory for speed)

References
----------
[1] J.B. Allen and D.A. Berkley, "Image method for efficiently simulating
    small-room acoustics," JASA, 65(4), April 1979.
[2] P.M. Peterson, "Simulating the response of multiple microphones to a
    single acoustic source in a reverberant room," JASA, 80(5), Nov. 1986.
"""

from __future__ import division

import math

import numpy as np
import scipy.signal
import torch

from . import mtype as _mtype

# ---------------------------------------------------------------------------
# Microphone polar-pattern parameter (rho) per type
# ---------------------------------------------------------------------------
_RHO = {
    ord("b"): 0.00,  # bidirectional
    ord("h"): 0.25,  # hypercardioid
    ord("c"): 0.50,  # cardioid
    ord("s"): 0.75,  # subcardioid
    ord("o"): 1.00,  # omnidirectional
}


# ---------------------------------------------------------------------------
# Vectorised microphone directivity
# ---------------------------------------------------------------------------

def _sim_microphone(Rp_x, Rp_y, Rp_z, angle, mtype_byte):
    """Compute directional gain for all image sources and microphones at once.

    Parameters
    ----------
    Rp_x, Rp_y, Rp_z : Tensor, shape (chunk, nMics)
        Displacement vector from each image source to each microphone,
        in normalised sample units.
    angle : ndarray, shape (2,)
        Microphone orientation [azimuth, elevation] in radians.
    mtype_byte : int
        ASCII code of the microphone-type character (e.g. ``ord('o')``).

    Returns
    -------
    gain : Tensor, shape (chunk, nMics)
    """
    if mtype_byte == ord("o"):
        return torch.ones_like(Rp_x)

    rho = _RHO.get(mtype_byte, 1.0)
    dist = torch.sqrt(Rp_x ** 2 + Rp_y ** 2 + Rp_z ** 2).clamp(min=1e-12)
    vartheta = torch.acos((Rp_z / dist).clamp(-1.0, 1.0))
    varphi = torch.atan2(Rp_y, Rp_x)
    az = float(angle[0])
    el = float(angle[1])
    gain = (
        math.sin(math.pi / 2.0 - el) * torch.sin(vartheta) * torch.cos(az - varphi)
        + math.cos(math.pi / 2.0 - el) * torch.cos(vartheta)
    )
    return rho + (1.0 - rho) * gain


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate(
    c,
    fs,
    r,
    s,
    L,
    beta=None,
    reverberation_time=None,
    nsample=None,
    mtype=_mtype.omnidirectional,
    order=-1,
    dim=3,
    orientation=None,
    hp_filter=True,
    device=None,
    chunk_size=4096,
):
    """Generate a room impulse response using the image method on GPU/CPU tensors.

    This is a drop-in replacement for ``rir_generator.generate()``.  All
    positional and keyword arguments are identical; two optional extras control
    the PyTorch backend.

    Parameters
    ----------
    c : float
        Sound velocity in m/s.
    fs : float
        Sampling frequency in Hz.
    r : array_like
        Receiver position(s), shape ``(3,)`` or ``(N, 3)``.
    s : array_like
        Source position, shape ``(3,)``.
    L : array_like
        Room dimensions ``[Lx, Ly, Lz]`` in m.
    beta : array_like, optional
        Reflection coefficients ``[βx1, βx2, βy1, βy2, βz1, βz2]``,
        shape ``(6,)`` or ``(3, 2)``.
    reverberation_time : float, optional
        Reverberation time T₆₀ in seconds.
    nsample : int, optional
        Number of output samples (default: ``T₆₀ * fs``).
    mtype : mtype, optional
        Microphone type (default: omnidirectional).
    order : int, optional
        Maximum reflection order; ``-1`` means unlimited.
    dim : int, optional
        Room dimensionality, ``2`` or ``3``.
    orientation : array_like, optional
        Microphone orientation ``[azimuth, elevation]`` in radians.
    hp_filter : bool, optional
        Apply Allen & Berkley 100 Hz high-pass filter (default: ``True``).
    device : str or torch.device, optional
        PyTorch device (default: CUDA if available, else CPU).
    chunk_size : int, optional
        Number of image sources processed per batch.  Larger values use more
        memory but reduce loop overhead; increase for powerful GPUs.

    Returns
    -------
    h : ndarray, shape (nsample, nMics)
        Room impulse response(s), matching the output of
        ``rir_generator.generate()``.
    """
    # ------------------------------------------------------------------
    # Input validation and beta / RT60 derivation
    # (mirrors __init__.py so this module is self-contained)
    # ------------------------------------------------------------------
    r = np.atleast_2d(np.asarray(r, dtype=np.float64)).T.copy()
    assert r.shape[0] == 3

    L = np.asarray(L, dtype=np.float64)
    assert L.shape == (3,)

    s = np.asarray(s, dtype=np.float64)
    assert s.shape == (3,)

    if beta is not None:
        beta = np.asarray(beta, dtype=np.float64)
        assert beta.shape == (6,) or beta.shape == (3, 2)
        beta = beta.reshape(3, 2)

    if (r > L[:, None]).any() or (r < 0).any():
        raise ValueError("r is outside the room")

    if (s > L).any() or (s < 0).any():
        raise ValueError("s is outside the room")

    if orientation is None:
        orientation = np.zeros(2, dtype=np.float64)
    orientation = np.atleast_1d(np.asarray(orientation, dtype=np.float64))
    if orientation.shape == (1,):
        orientation = np.pad(orientation, (0, 1), "constant")
    assert orientation.shape == (2,)

    assert order >= -1
    assert dim in (2, 3)

    V = np.prod(L)
    A = L[::-1] * np.roll(L[::-1], 1)

    if beta is not None:
        alpha = np.sum(np.sum(1 - beta ** 2, axis=1) * np.sum(A))
        reverberation_time = max(
            24 * np.log(10.0) * V / (c * alpha),
            0.128,
        )
    elif reverberation_time is not None:
        if reverberation_time != 0:
            S = 2 * np.sum(A)
            alpha = 24 * np.log(10.0) * V / (c * S * reverberation_time)
            if alpha > 1:
                raise ValueError(
                    "Error: The reflection coefficients cannot be "
                    "calculated using the current room parameters, "
                    "i.e. room size and reverberation time. Please "
                    "specify the reflection coefficients or change the "
                    "room parameters."
                )
            beta = np.full((3, 2), fill_value=np.sqrt(1 - alpha), dtype=np.float64)
        else:
            beta = np.zeros((3, 2), dtype=np.float64)
    else:
        raise ValueError(
            "Error: Specify either RT60 (ex: reverberation_time=0.4) or "
            "reflection coefficients (beta=[0.3,0.2,0.5,0.1,0.1,0.1])"
        )

    if nsample is None:
        nsample = int(reverberation_time * fs)

    if dim == 2:
        beta[-1, :] = 0.0

    # ------------------------------------------------------------------
    # Device and precision
    # ------------------------------------------------------------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    dtype = torch.float64

    # ASCII code of the microphone type character
    mtype_byte = mtype.value[0] if isinstance(mtype.value, (bytes, bytearray)) else ord(mtype.value)

    # ------------------------------------------------------------------
    # Normalise positions: convert metres → sample-distance units
    # ------------------------------------------------------------------
    cTs = c / fs
    beta_flat = beta.ravel()  # shape (6,)

    s_n = torch.tensor(s / cTs, dtype=dtype, device=device)          # (3,)
    L_n = torch.tensor(L / cTs, dtype=dtype, device=device)          # (3,)
    r_n = torch.tensor(r / cTs, dtype=dtype, device=device)          # (3, nMics)
    beta_t = torch.tensor(beta_flat, dtype=dtype, device=device)     # (6,)

    nMics = r_n.shape[1]

    # Hann-windowed sinc width: 2 * ROUND(0.004 * fs)
    # ROUND(x) = floor(x + 0.5) for x >= 0  (matches C++ macro exactly)
    Tw = 2 * int(math.floor(0.004 * fs + 0.5))

    # ------------------------------------------------------------------
    # Image-source search bounds (same formula as C++)
    # ------------------------------------------------------------------
    n1 = int(math.ceil(nsample / (2.0 * L_n[0].item())))
    n2 = int(math.ceil(nsample / (2.0 * L_n[1].item())))
    n3 = int(math.ceil(nsample / (2.0 * L_n[2].item())))

    # Grid indices, kept as int32 to save GPU memory
    mx_r = torch.arange(-n1, n1 + 1, dtype=torch.int32, device=device)
    my_r = torch.arange(-n2, n2 + 1, dtype=torch.int32, device=device)
    mz_r = torch.arange(-n3, n3 + 1, dtype=torch.int32, device=device)
    q_r = torch.arange(0, 2, dtype=torch.int32, device=device)
    j_r = torch.arange(0, 2, dtype=torch.int32, device=device)
    k_r = torch.arange(0, 2, dtype=torch.int32, device=device)

    # Flatten all combinations → shape (N_img,)
    MX, MY, MZ, Q, J, K = torch.meshgrid(
        mx_r, my_r, mz_r, q_r, j_r, k_r, indexing="ij"
    )
    MX = MX.reshape(-1)
    MY = MY.reshape(-1)
    MZ = MZ.reshape(-1)
    Q = Q.reshape(-1)
    J = J.reshape(-1)
    K = K.reshape(-1)
    N_img = MX.shape[0]

    # Tap offsets for the sinc window: [0, 1, ..., Tw-1]
    tap_idx = torch.arange(Tw, dtype=torch.int64, device=device)  # (Tw,)

    # Output buffer, mic-major layout: flat_idx = mic * nsample + sample
    # Reshaped to (nMics, nsample) and transposed to (nsample, nMics) at the end.
    flat_imp = torch.zeros(nMics * nsample, dtype=dtype, device=device)
    mic_offsets = (
        torch.arange(nMics, dtype=torch.int64, device=device) * nsample
    )  # (nMics,)

    # ------------------------------------------------------------------
    # Process image sources in chunks to bound peak GPU memory usage.
    # Peak memory per chunk ≈ chunk_size × nMics × Tw × 8 bytes (float64).
    # ------------------------------------------------------------------
    for cs in range(0, N_img, chunk_size):
        ce = min(cs + chunk_size, N_img)

        # Integer grid indices for this chunk
        mx_i = MX[cs:ce]
        my_i = MY[cs:ce]
        mz_i = MZ[cs:ce]
        q_i = Q[cs:ce]
        j_i = J[cs:ce]
        k_i = K[cs:ce]

        # Float versions for arithmetic
        mx_f = mx_i.to(dtype)
        my_f = my_i.to(dtype)
        mz_f = mz_i.to(dtype)
        q_f = q_i.to(dtype)
        j_f = j_i.to(dtype)
        k_f = k_i.to(dtype)

        # ----------------------------------------------------------------
        # Displacement vectors from each image source to each microphone.
        # Rp_x[i, m] = (1-2q)*s_x - r_x[m] + 2*mx*Lx   (sample units)
        # Shape: (chunk, nMics)
        # ----------------------------------------------------------------
        Rp_x = (
            (1.0 - 2.0 * q_f)[:, None] * s_n[0]
            - r_n[0][None, :]
            + (2.0 * mx_f * L_n[0])[:, None]
        )
        Rp_y = (
            (1.0 - 2.0 * j_f)[:, None] * s_n[1]
            - r_n[1][None, :]
            + (2.0 * my_f * L_n[1])[:, None]
        )
        Rp_z = (
            (1.0 - 2.0 * k_f)[:, None] * s_n[2]
            - r_n[2][None, :]
            + (2.0 * mz_f * L_n[2])[:, None]
        )
        dist = torch.sqrt(Rp_x ** 2 + Rp_y ** 2 + Rp_z ** 2)  # (chunk, nMics)

        # ----------------------------------------------------------------
        # Reflection coefficient product for each image source.
        # refl[i] = β₀^|mx-q| · β₁^|mx| · β₂^|my-j| · β₃^|my|
        #           · β₄^|mz-k| · β₅^|mz|
        # Shape: (chunk,) — independent of microphone position.
        # ----------------------------------------------------------------
        abs_mx_q = torch.abs(mx_i - q_i).to(dtype)
        abs_mx = torch.abs(mx_i).to(dtype)
        abs_my_j = torch.abs(my_i - j_i).to(dtype)
        abs_my = torch.abs(my_i).to(dtype)
        abs_mz_k = torch.abs(mz_i - k_i).to(dtype)
        abs_mz = torch.abs(mz_i).to(dtype)

        refl = (
            beta_t[0] ** abs_mx_q * beta_t[1] ** abs_mx
            * beta_t[2] ** abs_my_j * beta_t[3] ** abs_my
            * beta_t[4] ** abs_mz_k * beta_t[5] ** abs_mz
        )  # (chunk,)

        # ----------------------------------------------------------------
        # Validity mask: reflection order and distance bounds.
        # Shape: (chunk, nMics)
        # ----------------------------------------------------------------
        fdist = torch.floor(dist)  # (chunk, nMics)
        ord_sum = (
            torch.abs(2.0 * mx_f - q_f)
            + torch.abs(2.0 * my_f - j_f)
            + torch.abs(2.0 * mz_f - k_f)
        )  # (chunk,)
        valid = (
            ((ord_sum <= order) | (order == -1))[:, None]
            & (fdist < nsample)
        )  # (chunk, nMics)

        # ----------------------------------------------------------------
        # Microphone directivity gain.
        # Shape: (chunk, nMics)
        # ----------------------------------------------------------------
        mic_gain = _sim_microphone(Rp_x, Rp_y, Rp_z, orientation, mtype_byte)

        # ----------------------------------------------------------------
        # Aggregate gain: directivity × reflection product / spreading loss.
        # Equivalent to C++: gain = sim_microphone(…) * refl / (4π·dist·cTs)
        # Shape: (chunk, nMics); zero for invalid sources.
        # ----------------------------------------------------------------
        gain_f = torch.where(
            valid,
            mic_gain * refl[:, None] / (4.0 * math.pi * dist * cTs),
            torch.zeros(1, dtype=dtype, device=device),
        )

        # ----------------------------------------------------------------
        # Hann-windowed sinc interpolation kernel.
        #
        # t[i, m, n] = (n - Tw/2 + 1) - frac[i, m]
        # LPI[i, m, n] = 0.5·(1 + cos(2π·t/Tw)) · sinc(t)
        #
        # Sinc convention: C code uses sinc(x) = sin(x)/x, called as
        #   sinc(π·2·Fc·t) = sin(πt)/(πt)  with Fc = 0.5.
        # PyTorch: torch.sinc(x) = sin(πx)/(πx), so torch.sinc(t) matches. ✓
        # Shape: (chunk, nMics, Tw)
        # ----------------------------------------------------------------
        frac = dist - fdist  # fractional delay in samples, (chunk, nMics)
        t = (
            tap_idx[None, None, :].to(dtype)   # (1, 1, Tw)
            - 0.5 * Tw + 1.0
            - frac[:, :, None]                 # (chunk, nMics, 1)
        )  # (chunk, nMics, Tw)
        LPI = 0.5 * (1.0 + torch.cos(2.0 * math.pi * t / Tw)) * torch.sinc(t)

        # ----------------------------------------------------------------
        # Accumulate reflection contributions into the output buffer.
        #
        # For each valid (image source i, microphone m, tap n), add
        #   gain_f[i, m] * LPI[i, m, n]
        # at sample index  si[i, m, n] = fdist[i, m] - Tw//2 + 1 + n.
        # The flat buffer index is  mic_offsets[m] + si[i, m, n].
        # ----------------------------------------------------------------
        si = (
            fdist.long()[:, :, None] - Tw // 2 + 1
            + tap_idx[None, None, :]
        )  # (chunk, nMics, Tw)
        in_range = (si >= 0) & (si < nsample)  # (chunk, nMics, Tw)

        # Zero-out contributions that land outside the output window or
        # belong to invalid image sources.
        sv = torch.where(
            valid[:, :, None] & in_range,
            gain_f[:, :, None] * LPI,
            torch.zeros(1, dtype=dtype, device=device),
        )  # (chunk, nMics, Tw)

        # Map to flat buffer: index = mic_offset + sample_index
        flat_idx = (
            mic_offsets[None, :, None] + si.clamp(0, nsample - 1)
        ).reshape(-1)  # (chunk * nMics * Tw,)

        flat_imp.scatter_add_(0, flat_idx, sv.reshape(-1))

    # ------------------------------------------------------------------
    # Reshape buffer to (nsample, nMics) and move to numpy.
    # flat_imp layout: mic-major → flat_imp[m * nsample + s]
    # After reshape(nMics, nsample).T:  result[s, m]  ✓
    # ------------------------------------------------------------------
    imp_np = flat_imp.reshape(nMics, nsample).T.cpu().numpy()  # (nsample, nMics)

    # ------------------------------------------------------------------
    # High-pass filter (Allen & Berkley, 100 Hz, 2nd-order IIR).
    #
    # Transfer function:
    #   H(z) = (1 + A1·z⁻¹ + R1·z⁻²) / (1 - B1·z⁻¹ - B2·z⁻²)
    #
    # Applied to all microphones simultaneously via scipy sosfilt,
    # which is equivalent to the per-sample loop in the C++ code.
    # ------------------------------------------------------------------
    if hp_filter:
        W_hp = 2.0 * math.pi * 100.0 / fs
        R1 = math.exp(-W_hp)
        B1 = 2.0 * R1 * math.cos(W_hp)
        B2 = -R1 * R1
        A1 = -(1.0 + R1)
        sos = np.array([[1.0, A1, R1, 1.0, -B1, -B2]], dtype=np.float64)
        imp_np = scipy.signal.sosfilt(sos, imp_np, axis=0).astype(np.float64)

    return imp_np
