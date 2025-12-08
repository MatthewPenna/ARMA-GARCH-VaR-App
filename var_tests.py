import pandas as pd
import numpy as np
from arch import arch_model
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA 
import matplotlib.pyplot as plt
from temStaPy.distNTS import *
from scipy.stats import chi2, norm
import scipy.integrate as spi
from scipy.stats import t


def run_backtest(ticker, start_date, window_size, windows, VaR_quantile):
    """
    NTS ARMA GARCH:
    - Download price data from Yahoo Finance
    - Compute log returns * 1000 (like your original script)
    - Return a summary of what we did
    """

    # 1) Make sure the start_date is a string for yfinance
    if hasattr(start_date, "strftime"):
        start_str = start_date.strftime("%Y-%m-%d")
    else:
        start_str = str(start_date)

    # 2) Download data
    sp_data = yf.download(ticker, start = start_date, auto_adjust = True)["Close"]
    sp_ret = (pd.DataFrame(np.diff(np.log(sp_data.to_numpy()).T).T)* 1000 ).dropna() # multiplied by 1000 so the ARMA MLE will be more accurate
    sp_ret.columns = sp_data.columns
    sp_ret = sp_ret[ticker]

    n_returns = len(sp_ret)
    min_needed = window_size + windows - 1

    if n_returns < min_needed:
        return {
            "error": (
                f"Not enough data for chosen window settings. "
                f"Have {n_returns} returns but need at least {min_needed} "
                f"(window_size={window_size}, windows={windows})."
            ),
            "ticker": ticker,
            "start_date": start_str,
            "n_returns": int(n_returns),
        }
    violations_array = np.zeros(windows)
    comparison_returns = np.zeros(windows)
    VaR_array = np.zeros(windows)

    for window in range(windows):    
        print("WINDOW = ",window)
        # --- Get correct Window --- #
        new_sp_ret = sp_ret[window: window_size + window] # Current window

        #  ---- find model parameters using normal distribution ----   #

            # --- first estimate ARMA(1,1) ---   #
        arma_model = ARIMA(pd.DataFrame({ticker:new_sp_ret}), order = (1,0,1)) 
        arma_model = arma_model.fit() # estimates the parameters 
        arma_residuals = arma_model.resid # return - predicted value

            # --- second estimate GARCH(1,1) on residuals --- #
            
        garch_model = arch_model(arma_residuals, mean = "Zero" , vol = 'GARCH', p = 1 , q = 1, dist = 't').fit(update_freq=0, disp='off') # fit garch model on residuals using normal distribution. We assume residuals have mean = 0

        tomorrows_estimated_return = arma_model.forecast(steps=1).iloc[0] # mu t+1
        tomorrows_estimated_vol = np.sqrt(garch_model.forecast(horizon = 1).variance.iloc[-1, 0]) # sigma t+1
        standardized_residuals = garch_model.std_resid.values # These residuals should follow NTS. Standardized residuals represent iid pure shock term without accounting for conditional variance
        nts_params = fitstdnts(standardized_residuals)
        VaR = tomorrows_estimated_return + tomorrows_estimated_vol * VaRetnts(VaR_quantile, nts_params)* -1
        print(f"NTS quantile ={VaRetnts(VaR_quantile, nts_params):.3f}")
        VaR_array[window] = VaR
        comparison_returns[window] = sp_ret[window_size + window]
        if sp_ret[window_size + window] < VaR: # VaR violation
            violations_array[window] = 1 
        
    #--- Christofferson Test --- #

    alpha = 0.05

    n00=n01=n10=n11 = 0
    for i in range(1,windows):
        if(violations_array[i-1] == 0 and violations_array[i] == 0):
            n00 += 1
        elif(violations_array[i-1] == 0 and violations_array[i] == 1):
            n01 += 1
        elif(violations_array[i-1] == 1 and violations_array[i] == 0):
            n10 += 1
        else:
            n11 +=1


    print("Number of VaR violations:", sum(violations_array), "\nExpected number of violations: ", VaR_quantile * windows) #

    # --- Independence test --- #
    p01 = n01/(n00 +n01) # P(faliure at time t | no failure at time t-1)
    if n10 + n11 != 0: # The independce test cant occur if p11 is zero, so exclude that test if p11 = zero
        p11 = n11 / (n10 + n11) # P(failure at time t | failure at time t-1)
        pUC = (n01 + n11) / (n00 + n01+ n10 + n11) # P(failure at time t)


        # --- Test for independence of violations --- #
        LRind_numerator = (1-pUC) ** (n00 + n10) * pUC ** (n01+n11)
        LRind_denominator = (1-p01) ** n00 * p01 ** n01 * (1-p11) ** n10 *p11**n11
        LRind = -2 * np.log(LRind_numerator / LRind_denominator)

        # --- Test for correct number of violations --- #
        T = windows
        n = sum(violations_array)
        pi_hat = n/T
        LRuc_num = (1 - VaR_quantile)**(T - n) * VaR_quantile**n
        LRuc_den = (1 - pi_hat)**(T-n) * pi_hat ** n
        LRuc = -2 * np.log(LRuc_num / LRuc_den)

        # --- Get all p_values
        LR_cc = LRind + LRuc
        p_uc = 1 - chi2.cdf(LRuc, 1)
        p_ind = 1 - chi2.cdf(LRind, 1) 
        p_cc = 1 - chi2.cdf(LR_cc, 2) # This is the p-value of rejecting H0 that violations are independent and occur with probability alpha

        
        #(ChatGPT generated print statement)
        print(f"\n--- Christoffersen Likelihood Ratio VaR Backtests  ---")
        print(f"{'Test':<25}{'Statistic':>15}{'p-value':>15}{'df':>8}")
        print("-" * 63)
        print(f"{'Unconditional Coverage (LR_uc)':<25}{LRuc:>15.6f}{p_uc:>15.6g}{1:>8}")
        print(f"{'Independence (LR_ind)':<25}{LRind:>15.6f}{p_ind:>15.6g}{1:>8}")
        print(f"{'Conditional Coverage (LR_cc)':<25}{LR_cc:>15.6f}{p_cc:>15.6g}{2:>8}")
        print("-" * 63)


        print(f"Decision (alpha = {alpha}):")
        print(f"  LR_uc -> {'Reject H0' if p_uc < alpha else 'Do not reject H0'}")
        print(f"  LR_ind -> {'Reject H0' if p_ind < alpha else 'Do not reject H0'}")
        print(f"  LR_cc -> {'Reject H0' if p_cc < alpha else 'Do not reject H0'}") 

    else: # Only conduct the unconditional test
        T = windows
        n = sum(violations_array)
        pi_hat = n/T
        LRuc_num = (1 - VaR_quantile)**(T - n) * VaR_quantile**n
        LRuc_den = (1 - pi_hat)**(T-n) * pi_hat ** n
        LRuc = -2 * np.log(LRuc_num / LRuc_den)
        p_uc = 1 - chi2.cdf(LRuc, df=1)

        LR_ind = None
        p_ind = None
        LR_cc = None
        p_cc = None

        # Pretty print
        print("\n--- Christoffersen Unconditional Coverage Test For {ticker} from  - No conditional test due to zero violations ---")
        print(f"Sample size (T):          {T}")
        print(f"Observed violations (n):  {n}")
        print(f"Expected violations:      {VaR_quantile * T:.2f}")
        print(f"LR_uc statistic:          {LRuc:.6f}")
        print(f"p-value (df = 1):         {p_uc:.6g}")
        print(f"Decision (test alpha = {alpha}): "
            f"{'Reject H0' if p_uc < alpha else 'Do not reject H0'}")




    # 5) Build a result dictionary to send back to Streamlit

    result_NTS = {
        "ticker": ticker,
        "start_date": start_str,
        "window_size": int(window_size),
        "windows": int(windows),
        "var_quantile": float(VaR_quantile),
        "n_returns": int(n_returns),
        "n_violations": int(n),
        "expected_violations": float(VaR_quantile * T),
        "VaR_array": VaR_array.tolist(),
        "comparison_returns": comparison_returns.tolist(),
        "violations_array": violations_array.tolist(),
        "LR_uc": float(LRuc),
        "p_uc": float(p_uc),
        "LR_ind": None if LRind is None else float(LRind),
        "p_ind": None if p_ind is None else float(p_ind),
        "LR_cc": None if LR_cc is None else float(LR_cc),
        "p_cc": None if p_cc is None else float(p_cc),
        # Little previews so you don't have to scroll forever in JSON
        "VaR_preview": (np.round(VaR_array[:5], 4)/1000.0).tolist(),
        "comparison_preview": np.round(comparison_returns[:5], 4).tolist(),
    }

   
    """
    t ARMA GARCH:
   """

    violations_array = np.zeros(windows)
    comparison_returns = np.zeros(windows)
    VaR_array = np.zeros(windows)

    for window in range(windows):    
        print("WINDOW = ",window)
    # --- Get correct Window --- #
        new_sp_ret = sp_ret[window: window_size + window] # Current window

        #  ---- find model parameters using normal distribution ----   #

            # --- first estimate ARMA(1,1) ---   #
        arma_model = ARIMA(pd.DataFrame({ticker:new_sp_ret}), order = (1,0,1)) 
        arma_model = arma_model.fit() # estimates the parameters 
        arma_residuals = arma_model.resid # return - predicted value

            # --- second estimate GARCH(1,1) on residuals --- #
            
        garch_model = arch_model(arma_residuals, mean = "Zero" , vol = 'GARCH', p = 1 , q = 1, dist = 't').fit(update_freq=0, disp='off') # fit garch model on residuals using normal distribution. We assume residuals have mean = 0

        tomorrows_estimated_return = arma_model.forecast(steps=1).iloc[0] # mu t+1
        tomorrows_estimated_vol = np.sqrt(garch_model.forecast(horizon = 1).variance.iloc[-1, 0]) # sigma t+1
        nu = float(garch_model.params['nu'])   # degrees of freedom estimated by the model
        quant = t.ppf(VaR_quantile, df=nu)  
        VaR = tomorrows_estimated_return + tomorrows_estimated_vol * quant
        print(f"quantile = {quant}:.3f")
        print("nu =", nu)
        VaR_array[window] = VaR
        comparison_returns[window] = sp_ret[window_size + window]
        if sp_ret[window_size + window] < VaR: # VaR violation
            violations_array[window] = 1 
        
    #--- Christofferson Test --- #

    alpha = 0.05

    n00=n01=n10=n11 = 0
    for i in range(1,windows):
        if(violations_array[i-1] == 0 and violations_array[i] == 0):
            n00 += 1
        elif(violations_array[i-1] == 0 and violations_array[i] == 1):
            n01 += 1
        elif(violations_array[i-1] == 1 and violations_array[i] == 0):
            n10 += 1
        else:
            n11 +=1


    print("Number of VaR violations:", sum(violations_array), "\nExpected number of violations: ", VaR_quantile * windows) #

    # --- Independence test --- #
    p01 = n01/(n00 +n01) # P(faliure at time t | no failure at time t-1)
    if n10 + n11 != 0: # The independce test cant occur if p11 is zero, so exclude that test if p11 = zero
        p11 = n11 / (n10 + n11) # P(failure at time t | failure at time t-1)
        pUC = (n01 + n11) / (n00 + n01+ n10 + n11) # P(failure at time t)


        # --- Test for independence of violations --- #
        LRind_numerator = (1-pUC) ** (n00 + n10) * pUC ** (n01+n11)
        LRind_denominator = (1-p01) ** n00 * p01 ** n01 * (1-p11) ** n10 *p11**n11
        LRind = -2 * np.log(LRind_numerator / LRind_denominator)

        # --- Test for correct number of violations --- #
        T = windows
        n = sum(violations_array)
        pi_hat = n/T
        LRuc_num = (1 - VaR_quantile)**(T - n) * VaR_quantile**n
        LRuc_den = (1 - pi_hat)**(T-n) * pi_hat ** n
        LRuc = -2 * np.log(LRuc_num / LRuc_den)

        # --- Get all p_values
        LR_cc = LRind + LRuc
        p_uc = 1 - chi2.cdf(LRuc, 1)
        p_ind = 1 - chi2.cdf(LRind, 1) 
        p_cc = 1 - chi2.cdf(LR_cc, 2) # This is the p-value of rejecting H0 that violations are independent and occur with probability alpha

        
        #(ChatGPT generated print statement)
        print(f"\n--- Christoffersen Likelihood Ratio VaR Backtests  ---")
        print(f"{'Test':<25}{'Statistic':>15}{'p-value':>15}{'df':>8}")
        print("-" * 63)
        print(f"{'Unconditional Coverage (LR_uc)':<25}{LRuc:>15.6f}{p_uc:>15.6g}{1:>8}")
        print(f"{'Independence (LR_ind)':<25}{LRind:>15.6f}{p_ind:>15.6g}{1:>8}")
        print(f"{'Conditional Coverage (LR_cc)':<25}{LR_cc:>15.6f}{p_cc:>15.6g}{2:>8}")
        print("-" * 63)


        print(f"Decision (alpha = {alpha}):")
        print(f"  LR_uc -> {'Reject H0' if p_uc < alpha else 'Do not reject H0'}")
        print(f"  LR_ind -> {'Reject H0' if p_ind < alpha else 'Do not reject H0'}")
        print(f"  LR_cc -> {'Reject H0' if p_cc < alpha else 'Do not reject H0'}") 

    else: # Only conduct the unconditional test
        T = windows
        n = sum(violations_array)
        pi_hat = n/T
        LRuc_num = (1 - VaR_quantile)**(T - n) * VaR_quantile**n
        LRuc_den = (1 - pi_hat)**(T-n) * pi_hat ** n
        LRuc = -2 * np.log(LRuc_num / LRuc_den)
        p_uc = 1 - chi2.cdf(LRuc, df=1)

        LR_ind = None
        p_ind = None
        LR_cc = None
        p_cc = None

        # Pretty print
        print("\n--- Christoffersen Unconditional Coverage Test For {ticker} from  - No conditional test due to zero violations ---")
        print(f"Sample size (T):          {T}")
        print(f"Observed violations (n):  {n}")
        print(f"Expected violations:      {VaR_quantile * T:.2f}")
        print(f"LR_uc statistic:          {LRuc:.6f}")
        print(f"p-value (df = 1):         {p_uc:.6g}")
        print(f"Decision (test alpha = {alpha}): "
            f"{'Reject H0' if p_uc < alpha else 'Do not reject H0'}")




    # 5) Build a result dictionary to send back to Streamlit

    result_t = {
        "ticker": ticker,
        "start_date": start_str,
        "window_size": int(window_size),
        "windows": int(windows),
        "var_quantile": float(VaR_quantile),
        "n_returns": int(n_returns),
        "n_violations": int(n),
        "expected_violations": float(VaR_quantile * T),
        "VaR_array": VaR_array.tolist(),
        "comparison_returns": comparison_returns.tolist(),
        "violations_array": violations_array.tolist(),
        "LR_uc": float(LRuc),
        "p_uc": float(p_uc),
        "LR_ind": None if LRind is None else float(LRind),
        "p_ind": None if p_ind is None else float(p_ind),
        "LR_cc": None if LR_cc is None else float(LR_cc),
        "p_cc": None if p_cc is None else float(p_cc),
        # Little previews so you don't have to scroll forever in JSON
        "VaR_preview": np.round(VaR_array[:5], 4).tolist(),
        "comparison_preview": np.round(comparison_returns[:5], 4).tolist(),
    }


    """
    Normal ARMA GARCH:
    """

    violations_array = np.zeros(windows)
    comparison_returns = np.zeros(windows)
    VaR_array = np.zeros(windows)

    for window in range(windows):    
        print("WINDOW = ",window)
        # --- Get correct Window --- #
        new_sp_ret = sp_ret[window: window_size + window] # Current window

        #  ---- find model parameters using normal distribution ----   #

            # --- first estimate ARMA(1,1) ---   #
        arma_model = ARIMA(pd.DataFrame({ticker:new_sp_ret}), order = (1,0,1)) 
        arma_model = arma_model.fit() # estimates the parameters 
        arma_residuals = arma_model.resid # return - predicted value

            # --- second estimate GARCH(1,1) on residuals --- #
            
        garch_model = arch_model(arma_residuals, mean = "Zero" , vol = 'GARCH', p = 1 , q = 1, dist = 'normal').fit(update_freq=0, disp='off') # fit garch model on residuals using normal distribution. We assume residuals have mean = 0

        tomorrows_estimated_return = arma_model.forecast(steps=1).iloc[0] # mu t+1
        tomorrows_estimated_vol = np.sqrt(garch_model.forecast(horizon = 1).variance.iloc[-1, 0]) # sigma t+1

        VaR = tomorrows_estimated_return + tomorrows_estimated_vol * norm.ppf(VaR_quantile)
        VaR_array[window] = VaR
        comparison_returns[window] = sp_ret[window_size + window]
        print(f"Normal Quantile = {norm.ppf(VaR_quantile):.3f}")
        if sp_ret[window_size + window] < VaR: # VaR violation
            violations_array[window] = 1 
    #--- Christofferson Test --- #

    alpha = 0.05

    n00=n01=n10=n11 = 0
    for i in range(1,windows):
        if(violations_array[i-1] == 0 and violations_array[i] == 0):
            n00 += 1
        elif(violations_array[i-1] == 0 and violations_array[i] == 1):
            n01 += 1
        elif(violations_array[i-1] == 1 and violations_array[i] == 0):
            n10 += 1
        else:
            n11 +=1


    print("Number of VaR violations:", sum(violations_array), "\nExpected number of violations: ", VaR_quantile * windows) #

    # --- Independence test --- #
    p01 = n01/(n00 +n01) # P(faliure at time t | no failure at time t-1)
    if n10 + n11 != 0: # The independce test cant occur if p11 is zero, so exclude that test if p11 = zero
        p11 = n11 / (n10 + n11) # P(failure at time t | failure at time t-1)
        pUC = (n01 + n11) / (n00 + n01+ n10 + n11) # P(failure at time t)


        # --- Test for independence of violations --- #
        LRind_numerator = (1-pUC) ** (n00 + n10) * pUC ** (n01+n11)
        LRind_denominator = (1-p01) ** n00 * p01 ** n01 * (1-p11) ** n10 *p11**n11
        LRind = -2 * np.log(LRind_numerator / LRind_denominator)

        # --- Test for correct number of violations --- #
        T = windows
        n = sum(violations_array)
        pi_hat = n/T
        LRuc_num = (1 - VaR_quantile)**(T - n) * VaR_quantile**n
        LRuc_den = (1 - pi_hat)**(T-n) * pi_hat ** n
        LRuc = -2 * np.log(LRuc_num / LRuc_den)

        # --- Get all p_values
        LR_cc = LRind + LRuc
        p_uc = 1 - chi2.cdf(LRuc, 1)
        p_ind = 1 - chi2.cdf(LRind, 1) 
        p_cc = 1 - chi2.cdf(LR_cc, 2) # This is the p-value of rejecting H0 that violations are independent and occur with probability alpha

        
        #(ChatGPT generated print statement)
        print(f"\n--- Christoffersen Likelihood Ratio VaR Backtests  ---")
        print(f"{'Test':<25}{'Statistic':>15}{'p-value':>15}{'df':>8}")
        print("-" * 63)
        print(f"{'Unconditional Coverage (LR_uc)':<25}{LRuc:>15.6f}{p_uc:>15.6g}{1:>8}")
        print(f"{'Independence (LR_ind)':<25}{LRind:>15.6f}{p_ind:>15.6g}{1:>8}")
        print(f"{'Conditional Coverage (LR_cc)':<25}{LR_cc:>15.6f}{p_cc:>15.6g}{2:>8}")
        print("-" * 63)


        print(f"Decision (alpha = {alpha}):")
        print(f"  LR_uc -> {'Reject H0' if p_uc < alpha else 'Do not reject H0'}")
        print(f"  LR_ind -> {'Reject H0' if p_ind < alpha else 'Do not reject H0'}")
        print(f"  LR_cc -> {'Reject H0' if p_cc < alpha else 'Do not reject H0'}") 

    else: # Only conduct the unconditional test
        T = windows
        n = sum(violations_array)
        pi_hat = n/T
        LRuc_num = (1 - VaR_quantile)**(T - n) * VaR_quantile**n
        LRuc_den = (1 - pi_hat)**(T-n) * pi_hat ** n
        LRuc = -2 * np.log(LRuc_num / LRuc_den)
        p_uc = 1 - chi2.cdf(LRuc, df=1)

        LR_ind = None
        p_ind = None
        LR_cc = None
        p_cc = None

        # Pretty print
        print("\n--- Christoffersen Unconditional Coverage Test For {ticker} from  - No conditional test due to zero violations ---")
        print(f"Sample size (T):          {T}")
        print(f"Observed violations (n):  {n}")
        print(f"Expected violations:      {VaR_quantile * T:.2f}")
        print(f"LR_uc statistic:          {LRuc:.6f}")
        print(f"p-value (df = 1):         {p_uc:.6g}")
        print(f"Decision (test alpha = {alpha}): "
            f"{'Reject H0' if p_uc < alpha else 'Do not reject H0'}")




    # 5) Build a result dictionary to send back to Streamlit

    result_Normal = {
        "ticker": ticker,
        "start_date": start_str,
        "window_size": int(window_size),
        "windows": int(windows),
        "var_quantile": float(VaR_quantile),
        "n_returns": int(n_returns),
        "n_violations": int(n),
        "expected_violations": float(VaR_quantile * T),
        "VaR_array": VaR_array.tolist(),
        "comparison_returns": comparison_returns.tolist(),
        "violations_array": violations_array.tolist(),
        "LR_uc": float(LRuc),
        "p_uc": float(p_uc),
        "LR_ind": None if LRind is None else float(LRind),
        "p_ind": None if p_ind is None else float(p_ind),
        "LR_cc": None if LR_cc is None else float(LR_cc),
        "p_cc": None if p_cc is None else float(p_cc),
        # Little previews so you don't have to scroll forever in JSON
        "VaR_preview": np.round(VaR_array[:5], 4).tolist(),
        "comparison_preview": np.round(comparison_returns[:5], 4).tolist(),
    }

    return result_NTS, result_t, result_Normal