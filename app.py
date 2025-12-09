import streamlit as st
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from var_tests import run_backtest   

st.title("ARMA–GARCH Value at Risk Forecaster App")

st.write("Choose inputs, then compare ARMA-GARCH Models  with assumed Normal, t, and Normal Tempered Stable Distribution Innovations.")

# ---------- Sidebar inputs ----------
st.sidebar.header("Inputs")

ticker = st.sidebar.text_input(
    "What stock ticker do you want to analyze? (e.g. ^GSPC, AAPL)",
    "NVDA"
)

start_date = st.sidebar.date_input(
    "Start date for data",
    value=date(2017, 1, 1)
)

window_size = st.sidebar.number_input(
    "Rolling window size (days)",
    min_value=50,
    max_value=1000,
    value=700,
    step=10
)

windows = st.sidebar.number_input(
    "Number of windows",
    min_value=10,
    max_value=1000,
    value=250,
    step=10
)

VaR_quantile = st.sidebar.number_input(
    "VaR quantile",
    min_value=0.001,
    max_value=0.1,
    value=0.01,
    step=0.01
)

# ---------- Show current values ----------
st.subheader("Current input values")

st.write(f"**Ticker:** {ticker if ticker else '(none entered yet)'}")
st.write(f"**Start date:** {start_date}")
st.write(f"**Rolling window size:** {window_size} days")
st.write(f"**Number of windows:** {windows}")
st.write(f"**VaR quantile:** {VaR_quantile}")

st.markdown("---")

# ---------- Helper to display one model's results ----------
def show_model_results(model_name: str, res: dict, alpha: float = 0.05):
    """
    Nicely display Christoffersen test results for a single model
    (NTS, t, or Normal) using the structure in your dictionaries.
    """

    if res is None:
        st.error(f"{model_name}: No results returned.")
        return

    if "error" in res:
        st.error(f"{model_name}: {res['error']}")
        return

    st.markdown(f"### {model_name} model")
    st.write("**Christoffersen Likelihood Ratio Test**")

    # Basic counts
    T = res.get("T", len(res.get("VaR_array", [])))
    n_viol = res.get("n_violations", res.get("n", 0))
    expected = res.get("expected_violations", VaR_quantile * T)

    st.write(
        f"**Sample size (T):** {T}  \n"
        f"**Observed violations (n):** {n_viol}  \n"
        f"**Expected violations:** {expected:.2f}"
    )

    # Build table of LR stats (like your printout)
    rows = []

    LR_uc = res.get("LR_uc")
    p_uc = res.get("p_uc")
    rows.append({
        "Test": "Unconditional Coverage (LR_uc)",
        "Statistic": f"{LR_uc:.6f}" if LR_uc is not None else "—",
        "p-value": f"{p_uc:.6g}" if p_uc is not None else "—",
        "df": "1",
    })

    LR_ind = res.get("LR_ind")
    p_ind = res.get("p_ind")
    LR_cc = res.get("LR_cc")
    p_cc = res.get("p_cc")

    if LR_ind is not None or p_ind is not None:
        rows.append({
            "Test": "Independence (LR_ind)",
            "Statistic": f"{LR_ind:.6f}" if LR_ind is not None else "—",
            "p-value": f"{p_ind:.6g}" if p_ind is not None else "—",
            "df": "1",
        })

    if LR_cc is not None:
        rows.append({
            "Test": "Conditional Coverage (LR_cc)",
            "Statistic": f"{LR_cc:.6f}" if LR_cc is not None else "—",
            "p-value": f"{p_cc:.6g}" if p_cc is not None else "—",
            "df": "2",
        })

    df = pd.DataFrame(rows, columns=["Test", "Statistic", "p-value", "df"])
    st.table(df)

    # Decisions, matching your print statements
    st.write(f"**Decision (alpha = {alpha}):**")

    def decision(p):
        if p is None:
            return "Not applicable"
        return "Reject H₀" if p < alpha else "Do not reject H₀"

    st.write(f"- **LR_uc** → {decision(p_uc)}")

    if LR_ind is not None:
        st.write(f"- **LR_ind** → {decision(p_ind)}")
    else:
        st.write("- **LR_ind** → Not applicable (no 1→1 transitions)")

    if LR_cc is not None:
        st.write(f"- **LR_cc** → {decision(p_cc)}")
    else:
        st.write("- **LR_cc** → Not applicable")

    # ---- VaR vs Realized Returns Plot ----
    VaR_array = np.array(res.get("VaR_array", []))
    comparison_returns = np.array(res.get("comparison_returns", []))

    if VaR_array.size > 0 and comparison_returns.size == VaR_array.size:
        x = np.arange(VaR_array.size)

        # Boolean mask for violations: realized < VaR
        violations = comparison_returns < VaR_array

        fig, ax = plt.subplots(figsize=(10, 4))

        # Plot VaR
        ax.plot(x, VaR_array, label="Estimated VaR", linewidth=2)

        # Plot realized returns
        ax.plot(x, comparison_returns, label="Realized Return", linewidth=1)

        # Highlight VaR violations
        ax.scatter(
            x[violations],
            comparison_returns[violations],
            color="red",
            label="Violations",
            zorder=5,
        )

        ax.set_title(f"{model_name} VaR Backtesting: Estimated VaR vs Realized Returns")
        ax.set_xlabel("Window Number")
        ax.set_ylabel("log return x 1000")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()

        st.pyplot(fig)
    else:
        st.info(f"No plot available for {model_name} (missing VaR or return data).")        


# ---------- Button to run backtest ---------- #
st.subheader("Run the backtests (This will take a few minutes)")
run_clicked = st.button("Run backtest")

if run_clicked:
    if not ticker:
        st.error("Please enter a stock ticker before running the backtest.")
    else:
        with st.spinner("Running ARMA–GARCH VaR backtests..."):
            nts_result, t_result, normal_result = run_backtest(
                ticker=ticker,
                start_date=start_date,
                window_size=int(window_size),
                windows=int(windows),
                VaR_quantile=float(VaR_quantile),
            )

        st.success("Backtests completed.")

        st.subheader(
            f"Christoffersen Likelihood Ratio VaR Backtests for "
            f"{ticker} starting {start_date.strftime('%Y-%m-%d')}"
        )

        st.markdown("---")
        show_model_results("NTS", nts_result)

        st.markdown("---")
        show_model_results("t", t_result)

        st.markdown("---")
        show_model_results("Normal", normal_result)

