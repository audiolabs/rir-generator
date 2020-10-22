from __future__ import division
import numpy as np
from enum import Enum
from . import rir


class mtype(Enum):
    """ Microphone type. """

    bidirectional = b"b"
    b = b"b"
    cardioid = b"c"
    c = b"c"
    subcardioid = b"s"
    s = b"s"
    hypercardioid = b"h"
    h = b"h"
    omnidirectional = b"o"
    o = b"o"


def generate(
    c,
    fs,
    r,
    s,
    L,
    beta=None,
    reverberation_time=None,
    nsample=None,
    mtype=mtype.omnidirectional,
    order=-1,
    dim=3,
    orientation=None,
    hp_filter=True,
):
    """Generate room impulse response.

    Parameters
    ----------
    c : float
        Sound velocity in m/s. Usually between 340 and 350.
    fs : float
        Sampling frequency in Hz.
    r : array_like
        1D or 2D array of floats, specifying the :code:`(x, y, z)` coordinates of the receiver(s)
        in m. Must be of shape :code:`(3,)` or :code:`(x, 3)` where :code:`x`
        is the number of receivers.
    s : array_like
        1D array of floats specifying the :code:`(x, y, z)` coordinates of the source in m.
    L : array_like
        1D array of floats specifying the room dimensions :code:`(x, y, z)` in m.
    beta : array_like, optional
        1D array of floats specifying the reflection coefficients

        .. code-block::

            [beta_x1, beta_x2, beta_y1, beta_y2, beta_z1, beta_z2]

        or

        .. code-block::

            [(beta_x1, beta_x2), (beta_y1, beta_y2), (beta_z1, beta_z2)]

        Must be of shape :code:`(6,)` or :code:`(3, 2)`.

        You must define **exactly one** of :attr:`beta` or
        :attr:`reverberation_time`.
    reverberation_time : float, optional
        Reverberation time (T_60) in seconds.

        You must define **exactly one** of :attr:`beta` or
        :attr:`reverberation_time`.
    nsample : int, optional
        number of samples to calculate, default is :code:`T_60 * fs`.
    mtype : mtype, optional
        Microphone type, one of :class:`mtype`.
        Defaults to :class:`mtype.omnidirectional`.
    order : int, optional
        Reflection order, default is :code:`-1`, i.e. maximum order.
    dim : int, optional
        Room dimension (:code:`2` or :code:`3`), default is :code:`3`.
    orientation : array_like, optional
        1D array direction in which the microphones are pointed, specified
        using azimuth and elevation angles (in radians), default is
        :code:`[0, 0]`.
    hp_filter : boolean, optional
        Enable high-pass filter, the high-pass filter is enabled by default.

    Returns
    -------
    h : array_like
        The room impulse response, shaped `(nsample, len(r))`

    Example
    -------

    >>> import rir_generator
    >>> h = rir_generator.generate(
    ...     c=340,
    ...     fs=16000,
    ...     r=[
    ...       [2, 1.5, 2],
    ...       [2, 1.5, 3]
    ...     ],
    ...     s=[2, 3.5, 2],
    ...     L=[5, 4, 6],
    ...     reverberation_time=0.4,
    ...     nsample=4096,
    ...     mtype=rir_generator.mtype.omnidirectional,
    ... )


    """
    r = np.atleast_2d(np.asarray(r, dtype=np.double)).T.copy()
    assert r.shape[0] == 3

    L = np.asarray(L, dtype=np.double)
    assert L.shape == (3,)

    s = np.asarray(s, dtype=np.double)
    assert s.shape == (3,)

    if beta is not None:
        beta = np.asarray(beta, dtype=np.double)
        assert beta.shape == (6,) or beta.shape == (3, 2)
        beta = beta.reshape(3, 2)

    if (r > L[:, None]).any() or (r < 0).any():
        raise ValueError("r is outside the room")

    if (s > L).any() or (s < 0).any():
        raise ValueError("s is outside the room")

    # Make sure orientation is a 2-element array, even if passed a single value
    if orientation is None:
        orientation = np.zeros(2, dtype=np.double)
    orientation = np.atleast_1d(np.asarray(orientation, dtype=np.double))
    if orientation.shape == (1,):
        orientation = np.pad(orientation, (0, 1), "constant")
    assert orientation.shape == (2,)

    assert order >= -1
    assert dim in (2, 3)

    # Volume of room
    V = np.prod(L)
    # Surface area of walls
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

            beta = np.full((3, 2), fill_value=np.sqrt(1 - alpha), dtype=np.double)
        else:
            beta = np.zeros((3, 2), dtype=np.double)
    else:
        raise ValueError(
            "Error: Specify either RT60 (ex: reverberation_time=0.4) or "
            "reflection coefficients (beta=[0.3,0.2,0.5,0.1,0.1,0.1])"
        )

    if nsample is None:
        nsample = int(reverberation_time * fs)

    if dim == 2:
        beta[-1, :] = 0

    numMics = r.shape[1]

    imp = np.zeros((nsample, numMics), dtype=np.double)

    p_imp = rir.ffi.cast("double*", rir.ffi.from_buffer(imp))
    p_r = rir.ffi.cast("double*", rir.ffi.from_buffer(r))
    p_s = rir.ffi.cast("double*", rir.ffi.from_buffer(s))
    p_L = rir.ffi.cast("double*", rir.ffi.from_buffer(L))
    p_beta = rir.ffi.cast("double*", rir.ffi.from_buffer(beta))
    p_orientation = rir.ffi.cast("double*", rir.ffi.from_buffer(orientation))

    rir.lib.computeRIR(
        p_imp,
        float(c),
        float(fs),
        p_r,
        numMics,
        nsample,
        p_s,
        p_L,
        p_beta,
        mtype.value,
        order,
        p_orientation,
        1 if hp_filter else 0,
    )
    return imp
