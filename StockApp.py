import numpy as np
import streamlit as st
import datetime as dt
import pandas as pd
import alphavantage.av as av
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from optport import mv_solver
from sdes import MultiGbm
import finnhub
import time


def update_with_quotes(S):
    """
    Download the latest yahoo quotes
    :param S: stock prices to update
    :return: DataFrame of stock prices
    """
    # Get today's date
    today = dt.date.today()
    symbols = S.columns
    # Create a list to store the quotes
    quotes = []
    finnhub_client = finnhub.Client(api_key=st.secrets["FINNHUB_KEY"])
    # Loop over each ticker and get the current price quote
    for symbol in symbols:
        # V10 just broke as of 7/27/2023
        # quote = av.get_yahoo_quote_v10(symbol)
        # V6 works as of 7/27/2023; V6 brok as of 11/13/2023
        # quote = av.get_yahoo_quote_v6(symbol)
        quote = finnhub_client.quote(symbol)["c"]
        quotes.append(quote)
        time.sleep(1/len(symbols))

    # Create a new row with today's date and the quotes
    new_row = pd.DataFrame([quotes], columns=symbols, index=[today])
    st.write("Most recent quotes:")
    st.write(new_row)
    # Append the new row to the DataFrame S
    S = pd.concat([S, new_row])
    return S


# Define function to compute optimal allocations
def compute_allocations(X, gbm, ema_filter=0.0, timescale=1 / 252):
    """
    Compute the optimal allocations maximizing log-growth

    :param X: log-returns
    :param gbm: MultiGBM object
    :param ema_filter: ema_filter parameter
    :param timescale: time-scale of data
    :return: vector of allocations
    """
    gbm.fit(X, ema_filter=ema_filter, timescale=timescale)
    # st.write(gbm)
    w, g = mv_solver(gbm.drift, gbm.Sigma)
    mu = w.dot(gbm.drift)
    sigma = np.sqrt((w.T.dot(gbm.Sigma)).dot(w))
    return w, g, mu, sigma


# def mixture_allocations(X, gbm, timescale=1 / 252):
#     # Fit Gaussian mixture models with different numbers of components
#     bic_scores = []
#     for n_components in range(1, 11):
#         model = GaussianMixture(n_components=n_components, n_init=5, tol=10 ** -5, max_iter=200)
#         model.fit(X)
#         bic_scores.append(model.bic(X))
#     # Select the number of components with the lowest BIC score
#     best_n_components = np.argmin(bic_scores) + 1
#     st.write(f"Best number of components: {best_n_components}")
#     mix = GaussianMixture(n_components=best_n_components, n_init=5, tol=10 ** -5, max_iter=200)
#     mix.fit(X.to_numpy())
#     j = mix.predict(X.iloc[-1, :].to_numpy().reshape(1, -1))[0]
#     gbm.drift = mix.means_[j, :] / timescale
#     gbm.Sigma = mix.covariances_[j, :, :] / timescale
#     gbm.drift += np.diag(gbm.Sigma) / 2
#     w, g = mv_solver(gbm.drift, gbm.Sigma)
#     mu = w.dot(gbm.drift)
#     sigma = np.sqrt((w.T.dot(gbm.Sigma)).dot(w))
#     return w, g, mu, sigma


# Download data
# @st.cache(persist=True, allow_output_mutation=True)
def download_data(symbols):
    """
    Download stock data (daily adjusted close prices) for a given set of tickers/symbols
    :param symbols:
    :return:
    DataFrame of daily adjusted close prices
    """
    # Data parameters
    api = av.av()

    av_key = st.secrets["AV_KEY"]
    api.log_in(av_key)

    period = "daily"
    interval = None
    adjusted = True
    what = "adjusted_close"
    asset_type = "stocks"
    # 1. Get timescale and data
    timescale = av.timescale(period, interval, asset_type)
    data = api.get_assets(symbols, period, interval, adjusted, what)
    # Get YAHOO quotes:
    now = dt.datetime.now().astimezone(
        dt.timezone(dt.timedelta(hours=-4)))  # get current time in EDT timezone
    start_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    end_time = now.replace(hour=18, minute=0, second=0, microsecond=0)
    is_market_hours = now.weekday() < 5 and start_time <= now <= end_time


    is_market_hours = True

    if is_market_hours:
        data = update_with_quotes(data)
    else:
        st.write("Previous Close Prices:")
        st.write(data.iloc[-1, :])
    X = data.apply(lambda x: np.diff(np.log(x)))
    return X, timescale


if __name__ == "__main__":
    # Fit Multivariate GBM model
    gbm = MultiGbm()

    # Streamlit app code
    st.title("Optimal Log Growth Allocations")
    # Allow the user to enter a list of tickers
    default_ticker_list = "SPY, UBER, BB, RIOT, MULN, NKLA, COIN, FCEL, BND, AAPL, ROKU, TSLA, " \
                          "VXX, NVO, NIO, GOTU, NCMI, NVDA, RUN, WEAT"
    ticker_list = st.text_input("Enter a comma-separated list of tickers", default_ticker_list)

    # Convert the ticker list to a list of strings
    symbols = [s.strip() for s in ticker_list.split(",")]
    X = None
    timescale = None
    # download_button = st.button("Download stocks")
    # if download_button:
    #     X, timescale = download_data(symbols)

    # Define widgets
    ema_filter = st.slider("Select the EMA filter parameter:", 0.0, 1.0, 0.07, 0.01)
    bankroll = st.number_input("99\% VaR dollar amount:", 0.01, 10.**9, 100.)
    allocate_button = st.button("Allocate")

    # Update allocations when user clicks the update button
    if allocate_button:
        if X is None:
            X, timescale = download_data(symbols)
        w, g, mu, sigma = compute_allocations(X, gbm, ema_filter, timescale)
        st.write("================================")
        st.write("EWMA-GBM Allocations:")
        for i, asset in enumerate(X.columns):
            if np.abs(w[i]) > 0.001:
                st.write(f"{asset}: {w[i]:.2%}")
        st.write("Optimal growth rate = " + str(round(g, 6)))
        VaR = norm.ppf(0.001, loc=(mu - 0.5 * sigma ** 2) * timescale, scale=sigma * np.sqrt(timescale))
        st.write("Annual Drift = " + str(round(mu, 4)))
        st.write("Annual Volatility = " + str(round(sigma, 4)))
        st.write("99.9% Daily Value at Risk = " + str(round(VaR, 4)))
        # Compute the dollar amounts
        total = -bankroll/VaR
        st.write("Dollar amounts to hold")
        for i, asset in enumerate(X.columns):
            if np.abs(w[i]) > 0.001:
                st.write(f"{asset}: {round(total*w[i], 2)}")

