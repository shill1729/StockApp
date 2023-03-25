import numpy as np


def ema(X, alpha=0.07):
    """ Exponentially weighted moving average

    Keyword arguments:
    X -- np.array or pandas time-series of the signal
    alpha -- float the smoothing parameter in the unit interval (0,1).

    Closer values to zero for the parameter alpha give estimates
    that converge to the basic sample mean.
    """
    N = X.shape[0]
    w = (1 - alpha) ** np.arange(0, N)
    w = w[::-1]
    C = alpha / (1 - (1 - alpha) ** N)
    mu = w.T.dot(X) * C
    return mu


def ema_std(X, alpha=0.07):
    mu = ema(X, alpha)
    return np.sqrt(ema((X - mu) ** 2, alpha))


def ewmc(X, alpha=0.07):
    """ Compute the exponentially weighted moving covarince matrix of a pandas dataframe/series
    of (log)-returns.
    """

    N = X.shape[0]
    # Center data with either naive estimate or ema-estimate of mean.
    a1 = X-X.apply(np.mean, 0)
    # a1 = X - ema(X, alpha)

    # Compute weights
    ws = (1 - alpha) ** (np.arange(0, N))
    ws = ws * (alpha / (1 - (1 - alpha) ** N))
    ws = ws[::-1]
    # Compute weighted sample covariance in matrix form
    C = ws * a1.T
    Sigma = C @ a1
    return Sigma
