import numpy as np

from scipy.interpolate import PchipInterpolator
from scipy.fft import fft, ifft

from .temstarDist import *


def chf_subTS(u, param):
    a = param[0]
    th = param[1]
    if len(param) >= 3:
        dt = param[2]
    else:
        dt = 1
    y = np.exp(
        dt * (-2 * th ** (1 - a / 2) / a * ((th - 1j * u) ** (a / 2) - th ** (a / 2)))
    )
    return y


def dsubTS(x, subtsparam):
    alpha = subtsparam[0]
    theta = subtsparam[1]
    if len(subtsparam) >= 3:
        t = subtsparam[2]
    else:
        t = 1
    param = np.array([alpha, t * theta])
    pdf = (1 / t) * pdf_fft(x / t, param, chf_subTS)
    return pdf


def psubTS(x, subtsparam):
    alpha = subtsparam[0]
    theta = subtsparam[1]
    if len(subtsparam) >= 3:
        t = subtsparam[2]
    else:
        t = 1
    param = np.array([alpha, t * theta])
    cdf_subts = cdf_fft_gil_pelaez(x / t, param, chf_subTS)
    return cdf_subts


def qsubTS(u, subtsparam, maxt=50, du=0.01):
    return ipsubTS(u, subtsparam, maxt, du)


def rsubTS(n, subtsparam, maxt=50, du=0.01):
    u = np.random.rand(n)
    r = ipsubTS(u, subtsparam, maxt, du)
    return r


def ipsubTS(u, subtsparam, maxt=10, du=0.01):
    arg = np.arange(du, maxt, du)
    c = psubTS(arg, subtsparam)
    cc = cleanup_cdf(c, arg)
    x = cc["x"][cc["x"] > 0]
    y = cc["y"][cc["y"] > 0]
    x = np.concatenate(([0], x))
    y = np.concatenate(([0], y))
    ic = PchipInterpolator(y, x)(u)
    return ic
