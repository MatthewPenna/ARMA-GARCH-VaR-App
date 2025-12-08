import numpy as np
import pandas as pd
import functools
import cmath

from scipy.stats.kde import gaussian_kde
from scipy.optimize import minimize
from statsmodels.distributions.empirical_distribution import ECDF

from .temstarDist import pdf_fft, cdf_fft_gil_pelaez, avar_numint, invcdf_fft_gil_pelaez


def cvarnts(eps, ntsparam):
    newparam = change_ntsparam2stdntsparam(ntsparam)
    varstd = -qnts(eps, newparam["stdparam"])
    cvarstd = avar_numint(eps, varstd, newparam["stdparam"], chf_stdNTS)
    cvar = -newparam["mu"] + newparam["sig"] * cvarstd
    return cvar


def VaRetnts(eps, ntsparam):
    newparam = change_ntsparam2stdntsparam(ntsparam)
    nparam = [
        newparam["stdparam"][0],
        newparam["stdparam"][1],
        -newparam["stdparam"][2],
    ]
    varetstd = -qnts(eps, nparam)
    varet = newparam["mu"] + newparam["sig"] * varetstd
    return varet


def chf_NTS(u, param):
    a = param[0]
    th = param[1]
    b = param[2]
    g = param[3]
    m = param[4]
    if len(param) >= 6:
        dt = param[5]
    else:
        dt = 1

    y = np.exp(
        dt
        * (
            1j * (m - b) * u
            - 2
            * th ** (1 - a / 2)
            / a
            * (
                (th - 1j * (b * u + 1j * g**2 * u**2 / 2)) ** (a / 2)
                - th ** (a / 2)
            )
        )
    )

    return y


def chf_stdNTS(u, param):
    a = param[0]
    th = param[1]
    b = param[2]
    g = np.sqrt(abs(1 - b**2 * (2 - a) / (2 * th)))
    m = 0
    if len(param) >= 4:
        dt = param[3]
    else:
        dt = 1

    y = np.exp(
        dt
        * (
            1j * (m - b) * u
            - 2
            * th ** (1 - a / 2)
            / a
            * (
                (th - 1j * (b * u + 1j * g**2 * u**2 / 2)) ** (a / 2)
                - th ** (a / 2)
            )
        )
    )
    return y


def change_stdntsparam2ntsparam(stdparam, mu, sig, dt=1):
    a = stdparam[0]
    th = stdparam[1]
    bet = stdparam[2] * sig
    gam = sig * np.sqrt(1 - stdparam[2] ** 2 * (2 - a) / (2 * th))
    mu = mu
    ntsparam = np.array([a, th, bet, gam, mu, dt])
    ntsparam_names = ["alpha", "theta", "beta", "gamma", "mu", "dt"]
    ntsparam_dict = dict(zip(ntsparam_names, ntsparam))
    return ntsparam_dict


def change_ntsparam2stdntsparam(ntsparam):
    if len(ntsparam) == 3:
        a = ntsparam[0]  # alpha
        th_std = ntsparam[1]  # theta
        bet_std = ntsparam[2]  # beta
        sig_std = 1
        mu_std = 0
    else:
        if len(ntsparam) == 4:
            a = ntsparam[0]
            th = ntsparam[1] * ntsparam[3]
            bet = ntsparam[2] * ntsparam[3]
            gam = np.sqrt(ntsparam[3] - bet**2 * (2 - a) / (2 * th))
            m = 0
            dt = 1
        else:
            a = ntsparam[0]  # alpha
            th = ntsparam[1]  # theta
            bet = ntsparam[2]  # beta
            gam = ntsparam[3]  # gamma
            m = ntsparam[4]  # mu
            if len(ntsparam) >= 6:
                dt = ntsparam[5]
            else:
                dt = 1

        z = (2 - a) / (2 * th)
        th_std = th * dt
        bet_std = bet * np.sqrt(dt / (gam**2 + bet**2 * z))
        sig_std = np.sqrt(gam**2 * dt + bet**2 * dt * z)
        mu_std = m * dt

    stdntsparam = np.array([a, th_std, bet_std])
    param = {"stdparam": stdntsparam, "mu": mu_std, "sig": sig_std}
    return param


def dnts(xdata, ntsparam):
    newparam = change_ntsparam2stdntsparam(ntsparam)
    pdf = (1 / newparam["sig"]) * pdf_fft(
        (xdata - newparam["mu"]) / newparam["sig"],
        newparam["stdparam"],
        functools.partial(chf_stdNTS),
    )
    return pdf


def pnts(xdata, ntsparam, dz=2**-8, m=2**12):
    newparam = change_ntsparam2stdntsparam(ntsparam)
    cnts = cdf_fft_gil_pelaez(
        (xdata - newparam["mu"]) / newparam["sig"],
        newparam["stdparam"],
        chf_stdNTS,
        dz=2**-8,
        m=2**12,
    )
    return cnts


def ipnts(u, ntsparam, maxmin=[-10, 10], du=0.01):
    newparam = change_ntsparam2stdntsparam(ntsparam)

    x = invcdf_fft_gil_pelaez(
        u, newparam["stdparam"], chf_stdNTS, maxmin, du, dz=2**-8, m=2**12
    )

    y = x * newparam["sig"] + newparam["mu"]
    return y


def qnts(u, ntsparam):
    return ipnts(u, ntsparam)


def rnts(n, ntsparam, u=None):
    if u is None:
        u = np.random.rand(n)
    r = ipnts(u, ntsparam)
    return r


def moments_NTS(param):
    m4 = np.zeros(4, dtype=complex)
    cm = np.zeros(4, dtype=complex)

    al, th, b, gm, m = param
    cm[0] = m * 1j
    cm[1] = b**2 * th ** (1 - al / 2) * th ** (al / 2 - 2) * (al / 2 - 1) - gm**2
    cm[2] = (
        -(b**3)
        * th ** (1 - al / 2)
        * th ** (al / 2 - 3)
        * (al / 2 - 1)
        * (al / 2 - 2)
        * 1j
        + b * gm**2 * th ** (1 - al / 2) * th ** (al / 2 - 2) * (al / 2 - 1) * 3j
    )
    cm[3] = (
        6
        * b**2
        * gm**2
        * th ** (1 - al / 2)
        * th ** (al / 2 - 3)
        * (al / 2 - 1)
        * (al / 2 - 2)
        - 3 * gm**4 * th ** (1 - al / 2) * th ** (al / 2 - 2) * (al / 2 - 1)
        - b**4
        * th ** (1 - al / 2)
        * th ** (al / 2 - 4)
        * (al / 2 - 1)
        * (al / 2 - 2)
        * (al / 2 - 3)
    )

    cm /= 1j ** np.arange(1, 5)
    m4[0] = cm[0]
    m4[1] = cm[1]
    m4[2] = cm[2] / (cm[1] ** (3 / 2))
    m4[3] = cm[3] / (cm[1] ** 2)

    res = np.real(m4)
    res_names = ["mean", "variance", "skewness", "excess kurtosis"]
    return dict(zip(res_names, res))


def moments_stdNTS(param):
    a, th, bet = param
    gam = np.sqrt(abs(1 - bet**2 * (2 - a) / (2 * th)))
    m = 0
    m4 = moments_NTS([a, th, bet, gam, m])
    return m4


def sz(a, th):
    return np.sqrt((2 * th) / (2 - a))


def llhfnts(ntsparam, x, cemp, dispF=0):
    if len(ntsparam) == 3:
        ntsparam[2] = ntsparam[2] * sz(ntsparam[0], ntsparam[1])
    Fnts = pnts(x, ntsparam)  # Assuming pnts function is defined
    MSE = np.mean((cemp - Fnts) ** 2)

    if dispF == 1:
        print(",".join(str(e) for e in ntsparam) + ";" + str(MSE))

    return MSE


def pkden(q, kerncentres, lambda_val=None, bw=None, kernel="gaussian", lower_tail=True):
    if lambda_val is None and bw is None:
        bw = "scott"
    kde = gaussian_kde(kerncentres, bw_method=bw)
    if kernel != "gaussian":
        raise ValueError("Kernel other than 'gaussian' is not supported")
    cdf = np.array([kde.integrate_box_1d(-np.inf, qi) for qi in q])
    if not lower_tail:
        cdf = 1 - cdf
    return cdf


def fitnts(rawdat, initialparam=np.nan, maxeval=100, ksdensityflag=1):
    if np.isnan(initialparam).any():
        init = np.array([0.99, 2, 0])
    else:
        sp = change_ntsparam2stdntsparam(initialparam)
        init = sp["stdparam"]

    init[2] = init[2] / sz(init[0], init[1])

    mu = np.mean(rawdat)
    sig = np.std(rawdat)
    obs = (rawdat - mu) / sig

    if ksdensityflag == 1:
        ks = gaussian_kde(obs)
        x = ks.dataset[0]
        y = pkden(x, obs)
    else:
        Femp = ECDF(obs)
        x = np.linspace(min(obs), max(obs), num=1000)
        y = Femp(x)

    bounds = [(0.0001, 1.9999), (0.0001, 1000), (-0.9999, 0.9999)]
    res = minimize(
        lambda init: llhfnts(init, x, y),
        init,
        bounds=bounds,
        options={"maxiter": maxeval},
    )

    res.x[2] = res.x[2] * sz(res.x[0], res.x[1])

    retparam = change_stdntsparam2ntsparam(res.x, mu, sig, 1)
    keys = ["alpha", "theta", "beta", "gamma", "mu"]
    retparam = {key: retparam[key] for key in keys if key in retparam}

    return list(retparam.values())


def fitstdnts(rawdat, initialparam=np.nan, maxeval=100, ksdensityflag=1):
    if np.isnan(initialparam).any():
        init = np.array([0.99, 2, 0])
    else:
        init = initialparam

    init[2] = init[2] / sz(init[0], init[1])

    if ksdensityflag == 1:
        ks = gaussian_kde(rawdat)
        x = ks.dataset[0]
        y = pkden(x, rawdat)
    else:
        Femp = ECDF(rawdat)
        x = np.linspace(min(rawdat), max(rawdat), num=1000)
        y = Femp(x)

    bounds = [(0.0001, 1.9999), (0.0001, 1000), (-0.9999, 0.9999)]
    res = minimize(
        lambda init: llhfnts(init, x, y),
        init,
        bounds=bounds,
        options={"maxiter": maxeval},
    )

    res.x[2] = res.x[2] * sz(res.x[0], res.x[1])
    stdntsparam = [res.x[0], res.x[1], res.x[2]]
    return stdntsparam


def gensamplepathnts(npath, ntimestep, ntsparam, dt):
    if len(ntsparam) <= 4:
        param = ntsparam[0:3]
    elif len(ntsparam) >= 5:
        param = ntsparam[0:5]

    u = np.random.rand(npath, ntimestep)
    r = ipnts(u, param + [dt])
    x = np.cumsum(r, axis=1)

    colnames = np.arange(dt, dt * ntimestep + dt, dt)
    return pd.DataFrame(x, columns=colnames)
