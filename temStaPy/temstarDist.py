import numpy as np
import pandas as pd

from scipy.fftpack import fft
from scipy.interpolate import pchip
from scipy.integrate import quad


def cleanup_cdf(cdfret, argout):
    cdfret = np.array(cdfret)
    argout = np.array(argout)

    mask = cdfret > 0
    cdfret = cdfret[mask]
    argout = argout[mask]

    mask = cdfret < 1
    cdfret = cdfret[mask]
    argout = argout[mask]

    mask = np.diff(cdfret) != 0
    cdfret = cdfret[:-1][
        mask
    ]  # Exclude the last item because np.diff returns an array that has one fewer item
    argout = argout[:-1][mask]  # Exclude the last item

    while (
        len(cdfret[:-1][np.diff(cdfret) <= 0]) != 0
    ):  # Exclude the last item from cdfret when applying the mask
        mask = np.diff(cdfret) > 0
        cdfret = cdfret[:-1][mask]  # Exclude the last item
        argout = argout[:-1][mask]  # Exclude the last item

    c = pd.DataFrame({"x": argout, "y": cdfret})

    return c


def pdf_fft(arg, param, chf, h=2**-10, N=2**17):
    s = 1 / (h * N)
    t1 = np.arange(1, N + 1)
    t2 = 2 * np.pi * (t1 - 1 - N / 2) * s
    cfvalues = chf(t2, param)
    x1 = (-1) ** (t1 - 1) * cfvalues
    pdfft = np.real(fft(x1, overwrite_x=True)) * s * (-1) ** (t1 - 1 - N / 2)
    pdfft = np.where(pdfft >= 0, pdfft, 0)
    x = (t1 - 1 - N / 2) * h
    arg = np.sort(arg)
    f = pchip(x, pdfft)(arg)
    return f


def cdf_fft_gil_pelaez(arg, param, chf, dz=2**-12, m=2**17):
    k = np.arange(m)
    z = dz * (k + 0.5)
    x = np.pi * (2 * k - m + 1) / (m * dz)
    phi = chf(z, param)
    seq = (phi / z) * np.exp(1j * np.pi * (m - 1) / m * k)
    F_fft = 0.5 - (1 / np.pi) * np.imag(
        dz
        * np.exp(1j * np.pi * (m - 1) / (2 * m))
        * np.exp(-1j * np.pi / m * k)
        * fft(seq, overwrite_x=True)
    )
    F = pchip(x, F_fft)(arg)
    return F


def invcdf_fft_gil_pelaez(u, param, chf, maxmin, du, dz=2**-12, m=2**17):
    arg = np.arange(maxmin[0], maxmin[1], du)
    c = cdf_fft_gil_pelaez(arg, param, chf, dz, m)
    cc = cleanup_cdf(c, arg)
    y = pchip(cc["y"], cc["x"])(u)
    return y


def integrandavar(x, param, VaR, rho, chf):
    return np.real(
        np.exp(-1j * x * VaR) * chf(-x + rho * 1j, param) / (rho * 1j - x) ** 2
    )


def avar_numint(eps, VaR, param, chf, N=1000, rho=0.01):
    real_func = lambda x: np.real(integrandavar(x, param, VaR, rho, chf))
    f, _ = quad(real_func, 0, N)
    avar = VaR - np.exp(-VaR * rho) * f / (np.pi * eps)
    return avar
