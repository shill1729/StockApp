import numpy as np
import streamlit as st
import datetime as dt
import pandas as pd
import alphavantage.av as av
from scipy.stats import norm
from optport import mv_solver
from sdes import MultiGbm


def update_with_quotes(S, api):
    """
    Download the latest yahoo quotes
    :param S: stock prices to update
    :param api: api object
    :return: DataFrame of stock prices
    """
    # Get today's date
    today = dt.date.today()
    symbols = S.columns
    # Create a list to store the quotes
    quotes = []

    # Loop over each ticker and get the current price quote
    for symbol in symbols:
        quote = api.getYahooQuote(symbol)
        quotes.append(quote)

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
    w,g = mv_solver(gbm.drift, gbm.Sigma)
    mu = w.dot(gbm.drift)
    sigma = np.sqrt((w.T.dot(gbm.Sigma)).dot(w))
    return w, g, mu, sigma


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
    timescale = api.timescale(period, interval, asset_type)
    data = api.getAssets(symbols, period, interval, adjusted, what)
    # Get YAHOO quotes:
    now = dt.datetime.now().astimezone(
        dt.timezone(dt.timedelta(hours=-4)))  # get current time in EDT timezone
    start_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    end_time = now.replace(hour=18, minute=0, second=0, microsecond=0)
    is_market_hours = now.weekday() < 5 and start_time <= now <= end_time
    if is_market_hours:
        data = update_with_quotes(data, api)
    else:
        st.write("Previous Close Prices:")
        st.write(data.iloc[-1,:])
    X = data.apply(lambda x: np.diff(np.log(x)))
    return X, timescale


if __name__ == "__main__":
    # Fit Multivariate GBM model
    gbm = MultiGbm()

    # Streamlit app code
    st.title("Optimal Log Growth Allocations")
    # Allow the user to enter a list of tickers
    ticker_list = st.text_input("Enter a comma-separated list of tickers", "SPY,ROKU,NVDA,TSLA,GOTU,RIOT,TLT,VXX")

    # Convert the ticker list to a list of strings
    symbols = [s.strip() for s in ticker_list.split(",")]
    X = None
    timescale = None
    # download_button = st.button("Download stocks")
    # if download_button:
    #     X, timescale = download_data(symbols)

    # Define widgets
    ema_filter = st.slider("Select the EMA filter parameter:", 0.0, 1.0, 0.07, 0.01)
    allocate_button = st.button("Allocate")

    # Update allocations when user clicks the update button
    if allocate_button:
        if X is None:
            X, timescale = download_data(symbols)
        w,g,mu,sigma = compute_allocations(X, gbm, ema_filter, timescale)
        st.write("Optimal Allocations:")
        for i, asset in enumerate(X.columns):
            if np.abs(w[i]) > 0.001:
                st.write(f"{asset}: {w[i]:.2%}")
        st.write("Optimal growth rate = "+str(round(g, 6)))
        VaR = norm.ppf(0.001, loc=(mu-0.5*sigma**2)*timescale, scale=sigma*np.sqrt(timescale))
        st.write("Annual Drift = "+str(round(mu, 4)))
        st.write("Annual Volatility = " + str(round(sigma, 4)))
        st.write("99.9% Daily Value at Risk = "+str(round(VaR, 4)))
