#!/usr/bin/env python3
# −*− coding:utf-8 −*−

import numpy as np
from scipy.signal.windows import dpss


def pmtm(signal, dpss, axis=-1):
    '''
    Estimate the power spectral density of the input signal.
    signal: n-dimensional array of real or complex values
    dpss: the Slepian matrix
    axis:   axis along which to apply the Slepian windows. Default is the last one.
    '''
    # conversion to positive-only index
    axis_p = (axis + signal.ndim) % signal.ndim
    sig_exp_shape = list(signal.shape[:axis]) + [1] + list(signal.shape[axis:])
    tap_exp_shape = [1] * axis_p + \
        list(dpss.shape) + [1] * (signal.ndim - 1 - axis_p)
    signal_tapered = signal.reshape(
        sig_exp_shape) * dpss.reshape(tap_exp_shape)
    return np.fft.fftshift(np.mean(np.absolute(np.fft.fft(signal_tapered, axis=axis_p + 1))**2, axis=axis_p), axes=axis_p)


# ------------------------


if __name__ == '__main__':
    # a small test
    # Using traditional values used by Fritz, i.e. NW=4, Max_K = 2x NW-2 = 6
    mydpss = dpss(M=1024, NW=4, Kmax=6)
    sig = np.vectorize(complex)(np.random.rand(1024), np.random.rand(1024))
    print(pmtm(sig, mydpss))

    mydpss = dpss(M=128, NW=4, Kmax=6)
    sig = np.reshape(sig, (8, 128))
    print(pmtm(sig, mydpss, axis=1))
