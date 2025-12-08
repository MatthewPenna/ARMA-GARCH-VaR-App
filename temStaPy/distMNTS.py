import numpy as np
import pybobyqa

from scipy.stats import multivariate_normal
from scipy.integrate import quad, cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.stats.kde import gaussian_kde
from statsmodels.distributions.empirical_distribution import ECDF

from .distSubTS import rsubTS, ipsubTS, dsubTS
from .distNTS import pnts, dnts, qnts, sz, fitstdnts, cvarnts, chf_stdNTS


def randn(n, m):
    return np.random.normal(0, 1, (n, m))


def chol(a):
    return np.linalg.cholesky(a)


def rmnts(strPMNTS, numofsample, rW=None, rTau=None):
    if rW is None:
        rW = randn(numofsample, strPMNTS["ndim"])

    if rTau is None:
        rTau = rsubTS(numofsample, [strPMNTS["alpha"], strPMNTS["theta"]])

    N = strPMNTS["ndim"]
    beta = np.array(strPMNTS["beta"]).reshape(1, N)
    gamma = np.sqrt(1 - beta**2 * (2 - strPMNTS["alpha"]) / (2 * strPMNTS["theta"]))
    gamma = gamma.reshape(1, N)
    reps = np.dot(chol(strPMNTS["Rho"]).T, rW.T).T
    rTau = np.array(rTau).reshape(-1, 1)
    rstdmnts = ((rTau - 1) @ beta) + (np.sqrt(rTau) * (reps @ np.diag(gamma[0])))
    res_rmnts = rstdmnts @ np.diag(strPMNTS["sigma"]) + np.array(
        strPMNTS["mu"]
    ).reshape(1, N)

    return res_rmnts


def getGammaVec(alpha, theta, beta_vec):
    n = beta_vec.shape[0]  # number of rows in beta_vec
    gamma_vec = np.zeros((n, 1))
    for k in range(n):
        beta_k = beta_vec[k, :]  # kth row of beta_vec
        gamma_vec[k, 0] = np.sqrt(1 - beta_k**2 * (2 - alpha) / (2 * theta))
    return gamma_vec


def changeCovMtx2Rho(CovMtx, alpha, theta, betaVec, PDflag=True):
    n = len(betaVec)
    gammaVec = getGammaVec(alpha, theta, betaVec)
    Rho = CovMtx - (2 - alpha) / (2 * theta) * np.outer(betaVec, betaVec)
    igam = np.diag(1 / gammaVec.flatten())
    Rho = igam @ Rho @ igam
    if PDflag:
        Rho = nearestPD(Rho)
    return Rho


def nearestPD(A):
    """Find the nearest positive-definite matrix to input"""
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Check if matrix is positive definite"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def importantSampling(alpha, theta):
    u = np.concatenate(
        [
            np.linspace(0, 0.009, 10),
            np.linspace(0.01, 0.09, 9),
            np.linspace(0.1, 0.9, 9),
            [0.95, 0.99, 0.999, 0.9999, 0.99999, 0.999999],
        ]
    )
    ti = ipsubTS(u, [alpha, theta], maxt=20)
    subtsi = dsubTS(ti, [alpha, theta])
    return {"ti": ti, "subtsi": subtsi}


#def dMultiNorm_Subord(tVec, x, beta, sig0):
#    re = np.zeros(len(tVec))
#    for i, t in enumerate(tVec):
#        mu = beta * (t - 1)
#        Sig = t * sig0
#        re[i] = multivariate_normal.pdf(x, mean=mu, cov=Sig, allow_singular=True)
#    return re
def dMultiNorm_Subord(tVec, x, beta, sig0):
    if isinstance(tVec, float):
        tVec = np.array([tVec])
    re = np.zeros(len(tVec))
    for i, t in enumerate(tVec):
        mu = beta * (t - 1)
        Sig = t * sig0
        re[i] = multivariate_normal.pdf(x, mean=mu, cov=Sig)
    return re


def pmnts(x, st, subTS=None):
    xadj = (x - st["mu"]) / st["sigma"]
    return pMultiStdNTS(xadj, st, subTS)


def func_indegrand_cdf(t, x, st, ti, subtsi):
    Gt = pMultiNorm_Subord(t, x, st["alpha"], st["theta"], st["beta"], st["Rho"])
    ft = PchipInterpolator(ti, subtsi)(t)
    return Gt * ft


def pMultiNorm_Subord(tVec, x, alpha, theta, beta, rhoMtx):
    if isinstance(tVec, float):
        tVec = np.array([tVec])
    gamma = np.sqrt(1 - (2 - alpha) / (2 * theta) * beta**2)
    re = np.empty(len(tVec))
    for i, t in enumerate(tVec):
        if t == 0:
            re[i] = 0
        else:
            adjx = (x - beta * (t - 1)) / (gamma * np.sqrt(t))
            re[i] = multivariate_normal.cdf(adjx, mean=np.zeros(len(x)), cov=rhoMtx)
    return re


def pMultiStdNTS(x, st, subTS=None):
    if subTS is None:
        subTS = importantSampling(st["alpha"], st["theta"])

    ti = subTS["ti"]
    subtsi = subTS["subtsi"]

    # Integrate func_indegrand_cdf from 0 to max(ti)
    p = quad(func_indegrand_cdf, 0, max(ti), args=(x, st, ti, subtsi), limit=100)[0]
    p = np.clip(p, 0, 1)
    return p


def dmnts(x, st, subTS=None):
    xadj = (x - st["mu"]) / st["sigma"]
    return dMultiStdNTS(xadj, st, subTS) / np.prod(st["sigma"])


def dMultiStdNTS(x, st, subTS=None):
    if subTS is None:
        subTS = importantSampling(st["alpha"], st["theta"])

    ti = subTS["ti"]
    subtsi = subTS["subtsi"]

    gamma = np.sqrt(1 - (2 - st["alpha"]) / (2 * st["theta"]) * st["beta"] ** 2)
    sig0 = np.outer(gamma, gamma) * st["Rho"]

    d, _ = quad(
        func_indegrand, 0, max(ti), args=(x, st["beta"], sig0, ti, subtsi), limit=100
    )
    return d


def func_indegrand(t, x, beta, sig0, ti, subtsi):
    fe = dMultiNorm_Subord(t, x, beta, sig0)
    ft = PchipInterpolator(ti, subtsi)(t)
    return fe * ft


def pmarginalmnts(x, n, st):
    if n >= st["ndim"]:
        print("n must be less than st['ndim'].")
        return None
    if n < 0:
        print("n must be grater than or equal to 0.")
        return None
    ntsparam = [st["alpha"], st["theta"], st["beta"][n]]
    xdata = (x - st["mu"][n]) / st["sigma"][n]
    return pnts(xdata, ntsparam)


def dmarginalmnts(x, n, st):
    if n >= st["ndim"]:
        print("n must be less than st['ndim'].")
        return None
    if n < 0:
        print("n must be grater than or equal to 0.")
        return None
    xdata = (x - st["mu"][n]) / st["sigma"][n]
    ntsparam = [st["alpha"], st["theta"], st["beta"][n]]
    return dnts(xdata, ntsparam) / st["sigma"][n]


def qmarginalmnts(u, n, st):
    if n >= st["ndim"]:
        print("n must be less than st['ndim'].")
        return None
    if n < 0:
        print("n must be grater than or equal to 0.")
        return None
    ntsparam = [st["alpha"], st["theta"], st["beta"][n]]
    return st["sigma"][n] * qnts(u, ntsparam) + st["mu"][n]


def cvarmarginalmnts(eta, n, st):
    if n >= st["ndim"]:
        print("n must be less than st['ndim'].")
        return None
    if n < 0:
        print("n must be grater than or equal to 0.")
        return None
    ntsparam = [st["alpha"], st["theta"], st["beta"][n]]
    return st["sigma"][n] * cvarnts(eta, ntsparam) - st["mu"][n]
#def cvarmarginalmnts(eta, n, st):
#    if n > st["ndim"]:
#        print("n must be less than or equal to st['ndim'].")
#        return None
#    if n < 1:
#        print("n must be strictly positive integer.")
#        return None
#    ntsparam = [st["alpha"], st["theta"], st["beta"][n - 1]]
#    return st["sigma"][n - 1] * cvarnts(eta, ntsparam) - st["mu"][n - 1]

def fitmnts(returndata, n, alphaNtheta=None, stdflag=False, PDflag=True):
    strPMNTS = {
        "ndim": n,
        "mu": np.zeros((n, 1)),
        "sigma": np.ones((n, 1)),
        "alpha": 1,
        "theta": 1,
        "beta": np.zeros((n, 1)),
        "Rho": np.zeros((n, n)),
        "CovMtx": np.zeros((n, n)),
    }

    if stdflag:
        stdRetData = returndata
        strPMNTS["CovMtx"] = np.corrcoef(returndata, rowvar=False)
        #strPMNTS["CovMtx"] = np.corrcoef(returndata)
    else:
        strPMNTS["CovMtx"] = np.cov(returndata, rowvar=False)
        #strPMNTS["CovMtx"] = np.cov(returndata)
        strPMNTS["mu"] = np.mean(returndata, axis=0).reshape(n, 1)
        strPMNTS["sigma"] = np.sqrt(np.diag(strPMNTS["CovMtx"])).reshape(n, 1)
        stdRetData = (returndata - strPMNTS["mu"].T) / strPMNTS["sigma"].T

    athb = np.zeros((n, 3))
    if alphaNtheta is None:
        for k in range(n):
            stdntsparam = fitstdnts(stdRetData[:, k])
            athb[k, :] = stdntsparam
        alphaNtheta = np.mean(athb, axis=0)

    strPMNTS["alpha"] = alphaNtheta[0]
    strPMNTS["theta"] = alphaNtheta[1]

    betaVec = np.zeros((n, 1))
    for k in range(n):
        betaVec[k, 0] = fitstdntsFixAlphaThata(
            stdRetData[:, k], strPMNTS["alpha"], strPMNTS["theta"]
        )
    strPMNTS["beta"] = betaVec
    strPMNTS["Rho"] = changeCovMtx2Rho(
        np.cov(stdRetData, rowvar=False),
        #np.cov(stdRetData),
        strPMNTS["alpha"],
        strPMNTS["theta"],
        betaVec,
        PDflag,
    )

    strPMNTS["mu"] = np.squeeze(strPMNTS["mu"])
    strPMNTS["sigma"] = np.squeeze(strPMNTS["sigma"])
    strPMNTS["beta"] = np.squeeze(strPMNTS["beta"])
    return strPMNTS


def fitstdntsFixAlphaThata(
    rawdat, alpha, theta, initialparam=np.nan, maxeval=100, ksdensityflag=1
):
    init = np.zeros(1) if np.isnan(initialparam).any() else initialparam

    if ksdensityflag == 0:
        Femp = ECDF(rawdat)
        x = np.linspace(np.min(rawdat), np.max(rawdat), 1000)
        y = Femp(x)
    else:
        ks = gaussian_kde(rawdat)
        x = np.sort(ks.dataset.flatten())
        y = cumulative_trapezoid(ks(x), x, initial=0)

    def objective_func(betaparam, alpha=alpha, theta=theta, x=x, cemp=y, dispF=0):
        return llhfntsFixAlphaTheta(betaparam, alpha, theta, x, cemp, dispF)

    bounds = (np.array([-0.9999]), np.array([0.9999]))
    solver = pybobyqa.solve(
        objective_func,
        init,
        bounds=bounds,
        maxfun=maxeval,
        args=(alpha, theta, x, y, 0),
    )
    beta = solver.x[0] * sz(alpha, theta)

    return beta


def llhfntsFixAlphaTheta(betaparam, alpha, theta, x, cemp, dispF=0):
    betaparam *= sz(alpha, theta)
    Fnts = pnts(x, [alpha, theta, betaparam[0]])
    MSE = np.mean((cemp - Fnts) ** 2)
    if dispF == 1:
        print(f"{betaparam}, {MSE}")
    return MSE


def copulaStdNTS(u, st, subTS=None):
    x = np.empty(len(u))
    for j in range(len(u)):
        x[j] = qnts(u[j], [st["alpha"], st["theta"], st["beta"][j]])
    return pMultiStdNTS(x, st, subTS)


def dcopulaStdNTS(u, st, subTS=None):
    x = np.empty(len(u))
    y = 1
    for j in range(len(u)):
        x[j] = qnts(u[j], [st["alpha"], st["theta"], st["beta"][j]])
        y *= dnts(x[j], [st["alpha"], st["theta"], st["beta"][j]])
    return dMultiStdNTS(x, st, subTS) / y

def change_mvform2regform(stmnts_mv):
    n = stmnts_mv["ndim"]
    stmnts_reg = {
        "ndim": n,
        "alpha": 1,
        "theta": 1,
        "mu": np.zeros((n, 1)),
        "beta": np.zeros((n, 1)),
        "gamma": np.ones((n, 1)),
        "Rho": np.zeros((n, n)),
        "CovMtx": np.zeros((n, n)),
    }
    stmnts_reg["ndim"] = stmnts_mv["ndim"]
    stmnts_reg["alpha"] = stmnts_mv["alpha"]
    stmnts_reg["theta"] = stmnts_mv["theta"]
    stmnts_reg["mu"] = stmnts_mv["mu"]
    stmnts_reg["Rho"] = stmnts_mv["Rho"]
    stmnts_reg["CovMtx"] = stmnts_mv["CovMtx"]
    stmnts_reg["beta"]=stmnts_mv["sigma"]*stmnts_mv["beta"]
    gamma = np.sqrt(1-stmnts_mv["beta"]**2*(2-stmnts_mv["alpha"])/(2*stmnts_mv["theta"]))
    stmnts_reg["gamma"] = stmnts_mv["sigma"]*gamma
    return stmnts_reg

def change_regform2mvform(stmnts_reg):
    n = stmnts_reg["ndim"]
    stmnts_mv = {
        "ndim": n,
        "alpha": 1,
        "theta": 1,
        "mu": np.zeros((n, 1)),
        "sigma": np.ones((n, 1)),
        "beta": np.zeros((n, 1)),
        "Rho": np.zeros((n, n)),
        "CovMtx": np.zeros((n, n)),
    }
    stmnts_mv["ndim"] = stmnts_reg["ndim"]
    stmnts_mv["alpha"] = stmnts_reg["alpha"]
    stmnts_mv["theta"] = stmnts_reg["theta"]
    stmnts_mv["mu"] = stmnts_reg["mu"]
    stmnts_mv["Rho"] = stmnts_reg["Rho"]
    stmnts_mv["CovMtx"] = stmnts_reg["CovMtx"]
    stmnts_mv["sigma"] = np.sqrt(stmnts_reg["gamma"]**2+stmnts_reg["beta"]**2*(2-stmnts_reg["alpha"])/(2*stmnts_reg["theta"]))
    stmnts_mv["beta"] = stmnts_reg["beta"]/stmnts_mv["sigma"]
    return stmnts_mv
