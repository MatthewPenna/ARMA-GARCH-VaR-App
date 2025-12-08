# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 11:56:17 2025

@author: youngskim
"""
import numpy as np
from scipy.integrate import quad
from functools import partial

from .distNTS import ipnts, cvarnts, dnts
from .distMNTS import chf_stdNTS


def emp_CVaR_ret(alpha, ret):
    """
    Computes the empirical Conditional Value at Risk (CVaR) of a given return series.

    Parameters:
    alpha (float or list): Confidence level(s) (e.g., 0.95 for 95% confidence).
    ret (np.ndarray): Array of return values.

    Returns:
    dict: CVaR values for each confidence level in alpha.
    """
    eps = 1 - np.array(alpha)  # Convert confidence level to tail probability
    lossdata = -np.sort(ret)  # Sort losses in ascending order
    n = len(lossdata)
    CVaR = {}

    for j, e in enumerate(eps):
        cce = int(np.ceil(n * e))  # Index for VaR threshold
        CVaR[alpha[j]] = (np.sum(lossdata[cce:]) / n + ((cce - 1) / n - e) * lossdata[cce - 1]) / (1 - e)

    return CVaR



def mctStdDev(n, w, covMtx):
    """
    Computes the marginal contribution to standard deviation.

    Parameters:
    n (int): Index of the asset.
    w (np.ndarray): Portfolio weights (1D array).
    covMtx (np.ndarray): Covariance matrix.

    Returns:
    float: Marginal contribution to standard deviation.
    """
    sig = np.sqrt(w @ covMtx @ w.T)  # Portfolio standard deviation
    return (w @ covMtx[:, n]) / sig

def dBeta(n, w, betaArray, covMtx):
    """
    Computes the marginal contribution of beta.

    Parameters:
    n (int): Index of the asset.
    w (np.ndarray): Portfolio weights (1D array).
    betaArray (np.ndarray): Array of beta values for assets.
    covMtx (np.ndarray): Covariance matrix.

    Returns:
    float: Marginal contribution of beta.
    """
    barsig = np.sqrt(w @ covMtx @ w.T)  # Portfolio standard deviation
    barBeta = np.sum(w * betaArray)  # Weighted sum of betas
    mctsd = mctStdDev(n, w, covMtx)  # Marginal contribution to std dev
    sig = np.sqrt(covMtx[n, n])  # Standard deviation of asset n
    
    return sig * betaArray[n] / barsig - (barBeta / barsig) * mctsd

def psi_stdNTS(z, alpha, theta, beta):
    """
    Computes the psi function for the standard Normal Tempered Stable (NTS) distribution.

    Parameters:
    z (complex or ndarray): Input complex number.
    alpha (float): Stability parameter.
    theta (float): Scale parameter.
    beta (float): Skewness parameter.

    Returns:
    complex: Computed psi value.
    """
    sz = (2 - alpha) / (2 * theta)
    return (-1j * z + (1 - 1j * z * beta / theta + (1 - beta**2 * sz) * z**2 / (2 * theta))**(alpha / 2 - 1) * (1j * z + beta * sz * z**2))


def int_phi_psi(u, x, alpha, theta, beta, rho=0.1):
    """
    Computes the integrand function for dFdBeta_eta.

    Parameters:
    u (float): Integration variable.
    x (float): Input value.
    alpha (float): Stability parameter.
    theta (float): Scale parameter.
    beta (float): Skewness parameter.
    rho (float, optional): Decay parameter (default: 0.1).

    Returns:
    complex: Computed value of the integrand.
    """
    param = (alpha, theta, beta)
    res = np.exp(-1j * u * x) * chf_stdNTS(u + rho * 1j, param) / (rho - u * 1j)
    res *= psi_stdNTS(u + 1j * rho, alpha, theta, beta)
    return res

def dFdBeta_eta(x, eta, alpha, theta, beta, rho=0.1, N=20):
    """
    Computes dFdB using numerical integration.

    Parameters:
    x (float): Input value.
    eta (float): Unused parameter (exists in original R function).
    alpha (float): Stability parameter.
    theta (float): Scale parameter.
    beta (float): Skewness parameter.
    rho (float, optional): Decay parameter (default: 0.1).
    N (int, optional): Upper integration limit (default: 20).

    Returns:
    float: Computed dFdB value.
    """
    func = partial(int_phi_psi, x=x, alpha=alpha, theta=theta, beta=beta, rho=rho)

    # Perform numerical integration over [0, N]
    fn, _ = quad(lambda u: np.real(func(u)), 0, N)

    # Compute dFdB
    dFdB = np.real(fn) * np.exp(rho * x) / np.pi
    return dFdB

def dinvCdf_stdNTS_int(eta, x=None, alpha=None, theta=None, beta=None):
    """
    Computes the inverse cumulative distribution function (iCDF) derivative.

    Parameters:
    eta (float): Quantile level.
    x (float, optional): Input value (computed if None).
    alpha (float): Stability parameter.
    theta (float): Scale parameter.
    beta (float): Skewness parameter.

    Returns:
    float: Computed derivative of the inverse CDF.
    """
    if x is None:
        x = ipnts(eta, (alpha, theta, beta))  # ipnts needs to be implemented

    denom = dnts(x, (alpha, theta, beta))  # dnts needs to be implemented
    nom = dFdBeta_eta(x, eta, alpha, theta, beta)

    return -nom / denom

def mctVaR_MNTS(n, eta, w, stmnts, iCDFstd=None):
    """
    Computes the Marginal Contribution to VaR (MCT VaR) for MNTS.

    Parameters:
    n (int): Index of the asset.
    eta (float): Confidence level.
    w (np.ndarray): Portfolio weights.
    stmnts (dict): Dictionary containing parameters {mu, CovMtx, alpha, theta, beta}.
    iCDFstd (float, optional): Precomputed iCDF value.

    Returns:
    float: MCT VaR for the asset.
    """
    barsig = np.sqrt(w @ stmnts["CovMtx"] @ w.T)
    mcts = mctStdDev(n, w, stmnts["CovMtx"])  # Function mctStdDev needs to be implemented
    db = dBeta(n, w, stmnts["beta"], stmnts["CovMtx"])  # Function dBeta needs to be implemented
    barBeta = np.sum(w * stmnts["beta"])

    if iCDFstd is None:
        iCDFstd = ipnts(eta, (stmnts["alpha"], stmnts["theta"], barBeta))  # ipnts needs to be implemented

    dicdf = dinvCdf_stdNTS_int(eta, iCDFstd, stmnts["alpha"], stmnts["theta"], barBeta)
    return -stmnts["mu"][n] - iCDFstd * mcts - barsig * db * dicdf

def portfolio_VaR_CVaR_MNTS(eta, w, stmnts):
    """
    Computes portfolio VaR and CVaR for MNTS.

    Parameters:
    eta (float): Confidence level.
    w (np.ndarray): Portfolio weights.
    stmnts (dict): Dictionary containing parameters {mu, CovMtx, alpha, theta, beta}.

    Returns:
    dict: Dictionary with VaR and CVaR values.
    """
    barmu = np.sum(w * stmnts["mu"])
    barBeta = np.sum(w * stmnts["beta"])
    barsigma = np.sqrt(w @ stmnts["CovMtx"] @ w.T)

    VaR_stdNTS = -ipnts(eta, (stmnts["alpha"], stmnts["theta"], barBeta))  # ipnts needs to be implemented
    CVaR_stdNTS = cvarnts(eta, (stmnts["alpha"], stmnts["theta"], barBeta))  # cvarnts needs to be implemented

    VaR_Port_NTS = -barmu + barsigma * VaR_stdNTS
    CVaR_Port_NTS = -barmu + barsigma * CVaR_stdNTS

    return {"VaRNTS": VaR_Port_NTS, "CVaRNTS": CVaR_Port_NTS}


def integrand_dCVaRstdNTS(u, eta, alpha, theta, beta, rho=0.0001):
    """
    Computes the integrand function for dCVaRstdNTS numerical integration.

    Parameters:
    u (float): Integration variable.
    eta (float): Confidence level.
    alpha (float): Stability parameter.
    theta (float): Scale parameter.
    beta (float): Skewness parameter.
    cv (float, optional): Precomputed CVaR value.
    v (float, optional): Precomputed inverse CDF value.
    rho (float, optional): Small constant to ensure numerical stability (default: 0.0001).

    Returns:
    complex: Computed value of the integrand.
    """
    param = (alpha, theta, beta)

    # Compute inverse CDF if not provided
    Finv = ipnts(eta, param)  # ipnts needs to be implemented

    res = np.exp((1j * u + rho) * Finv) * chf_stdNTS(-u + rho * 1j, param)
    res *= psi_stdNTS(-u + 1j * rho, alpha, theta, beta)
    res *= (1 / (rho * 1j - u) ** 2)

    return res


def dCVaRstdNTS_numint(eta, alpha, theta, beta, N=20, rho=0.0001):
    """
    Computes the derivative of Conditional Value at Risk (CVaR) using numerical integration.

    Parameters:
    eta (float): Confidence level.
    alpha (float): Stability parameter.
    theta (float): Scale parameter.
    beta (float): Skewness parameter.
    N (int, optional): Upper integration limit (default: 20).
    rho (float, optional): Small constant for numerical stability (default: 0.0001).

    Returns:
    float: Computed derivative of CVaR.
    """
    func = partial(integrand_dCVaRstdNTS, eta=eta, alpha=alpha, theta=theta, beta=beta, rho=rho)
    
    fn, _ = quad(lambda u: np.real(func(u)), 0, N)

    dcvar = -fn / (np.pi * eta)
    return dcvar



def mctCVaR_MNTS(n, eta, w, stmnts, CVaRstd=None, dCVaRstd=None):
    """
    Computes the Marginal Contribution to CVaR (MCT CVaR) for MNTS.

    Parameters:
    n (int): Index of the asset. n is integer in [0, stmnts["ndim"]-1]
    eta (float): Confidence level.
    w (np.ndarray): Portfolio weights.
    stmnts (dict): Dictionary containing parameters {mu, CovMtx, alpha, theta, beta}.
    CVaRstd (float, optional): Precomputed standard CVaR.
    dCVaRstd (float, optional): Precomputed derivative of CVaR.
    
    Returns:
    float: MCT CVaR for the asset.
    """
    barBeta = np.sum(w * stmnts["beta"])

    if CVaRstd is None:
        CVaRstd = cvarnts(eta, (stmnts["alpha"], stmnts["theta"], barBeta))  # cvarnts needs to be implemented
    if dCVaRstd is None:
        dCVaRstd = dCVaRstdNTS_numint(eta, stmnts["alpha"], stmnts["theta"], barBeta)

    barsig = np.sqrt(w @ stmnts["CovMtx"] @ w.T)
    mcts = mctStdDev(n, w, stmnts["CovMtx"])  # mctStdDev needs to be implemented
    db = dBeta(n, w, stmnts["beta"], stmnts["CovMtx"])  # dBeta needs to be implemented

    return -stmnts["mu"][n] + CVaRstd * mcts + barsig * db * dCVaRstd

