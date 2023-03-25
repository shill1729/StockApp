import numpy as np
from emas import *
from optport import mv_solver
from optport import kelly_criterion as kc


def bm(x0, T, n):
    """ Standard one-dimensional Brownian motion in R^1.
    """
    b = np.random.normal(size=n)
    h = T / n
    b = np.hstack((0, np.sqrt(h) * b))
    b = np.cumsum(b)
    return b + x0


def hyperbolic_bm(x0, T, n):
    """ Standard Hyperbolic Brownian motion in the half-plane model of hyperbolic space.
    """
    b1 = bm(0, T, n)
    b2 = bm(0, T, n)
    t = np.linspace(0, T, n + 1)
    y = x0[1] * np.exp(b1 - 0.5 * t)
    x = x0[0] + x0[1] * np.cumsum(y * np.hstack((0, np.diff(b2))))
    return np.array([x, y]).T, np.array([b1, b2]).T + x0


def euler_maruyama(x0, T, mu, sigma, n):
    """ Assumes x0, mu and sigma are all scalars.
    """
    x = np.zeros(n + 1)
    x[0] = x0
    h = T / n
    for i in range(n):
        x[i + 1] = x[i] + mu(i * h, x[i]) * h + sigma(i * h, x[i]) * np.sqrt(h) * np.random.normal()
    return x


def euler_maruyama_2d(x0, T, mu, b, n, d=2, m=None):
    """ Assumes x0, mu are vectors and b is matrix such that bb^T is the covariance matrix.


    (Parameters)
    x0 : initial point, vector/array of m dimensions
    T : the time span of the simulation
    mu : the drift coefficient function, of time and space (t,x)
    b : the square root matrix of the covariance function of time and space (t,x)
    must be of dimensions m x d
    d : integer the dimensions of the Brownian motion driving the SDE
    m : the number of dimensions of the SDE system, optional
    """

    if m is None:
        m = x0.shape[0]
    h = T / n
    x = np.zeros((n + 1, m))
    x[0, :] = x0
    for i in range(n):
        x[i + 1, :] = x[i, :] + mu(i * h, x[i, :]) * h + b(i * h, x[i, :]).dot(
            np.random.normal(scale=np.sqrt(h), size=d))
    return x


def ensemble_average(ensemble, f=None):
    """ Compute the sample average of a function of the process over independent sample-paths, conditional on the
    initial point of the process.

    Parameters:
        ensemble: the array of sample-paths.
        f: the function to compute the expected value of conditional on initial point of the SDE
    """
    if f is None:
        f = lambda x: x
    # If ensemble has shape (N, n, d) where N is number of sample-paths, n is the number of
    # time steps, and d, the dimension of the process, then you can use
    # np.mean(..., axis=0) to sample-average over the ensemble at each time-step.
    return np.mean(ensemble, axis=0)


class Gbm(object):
    def __init__(self, mu=None, sigma=None, timescale=1.0 / 252.0):
        """ A class for a geometric Brownian motion. The SDE is defined by a drift coefficient
        and volatility coefficient.

        Attributes:
        mu        -- float, the drift coefficient, a real number
        sigma     -- float, the volatility coefficient, a positive real number
        timescale -- float, the timescale factor to multiply by when estimating parameters

        Methods:
        fit             -- fit a GBM to log returns
        kelly_fraction  -- compute the optimal growth-rate leverage
        browne_fraction -- compute the optimal chance-maximizer leverage of being above target
        sample_path     -- simulate a sample path of the specified GBM

        """
        self.mu = mu
        self.sigma = sigma
        self.timescale = timescale

    def __str__(self):
        return "Drift and Volatility = " + str((self.mu, self.sigma))

    def mean_growth(self):
        """ The buy-and-hold growth rate for the stock.
        """
        return self.mu - 0.5 * self.sigma ** 2

    def optimal_growth(self):
        """ The optimal growth rate achieved by using the Kelly-fraction.
        """
        return 0.5 * (self.mu / self.sigma) ** 2

    def fit(self, X, ema_filter=0):
        """ Fit a geometric Brownian motion to log-increment data. Optionally use EMA estimates for
        the drift and volatility.

        Keyword arguments:
        X          -- pandas series or numpy array, of univariate log-increments of prices
        ema_filter -- float, if zero naive sample estimates are used, if > 0, an EMA filter is used.
        """
        if ema_filter == 0.0:
            self.sigma = np.std(X) / np.sqrt(self.timescale)
            self.mu = np.mean(X) / self.timescale
            self.mu += 0.5 * self.sigma ** 2
        elif ema_filter > 0.0:
            self.mu = ema(X, ema_filter)
            self.sigma = np.sqrt(ema((X - self.mu) ** 2, ema_filter) / self.timescale)
            self.mu = self.mu / self.timescale + 0.5 * self.sigma ** 2

    def kelly_fraction(self, r=0.0):
        """ The classical kelly fraction mu/sigma^2
        """
        return (self.mu - r) / self.sigma ** 2

    def browne_fraction(self, t=0, x=1, b=1, r=0.0, T=1):
        """ The classical Browne-fraction, i.e. the alllocation that maximizes the chance
        of having a return above a price b by time T given wealth X_t=x at time t.
        """
        if x >= b * np.exp(-r * (T - t)):
            raise ValueError("Initial wealth must be less than discounted target: b*e^{-r(T-t)}")
        if x <= 0:
            raise ValueError("Initial wealth must be positive")
        if b <= 0:
            raise ValueError("Target wealth must be positive")
        if b < x * np.exp(r * (T - t)):
            raise ValueError("Target wealth must be greater than inital wealth with bond-return.")
        if b == x and r > 0:
            raise ValueError("the risk-free rate 'r' must be negative for target=initial.")
        return np.sqrt(np.log(b / x) - r * (T - t)) * np.sqrt(2 / (T - t)) * self.sigma

    def sample_path(self, s0, tn, n=1000):
        """ Generate a sample path using the exact solution of the SDE and a simulated
        brownian motion.

        Keyword arguments:
        s0 -- float, the initial spot price
        tn -- float, the time horizon
        n  -- int, the number of time sub-intervals in the simulation

        Returns:
        tuple of time and price
        """
        b = np.random.normal(size=n)
        h = tn / n
        b = np.hstack((0, np.sqrt(h) * b))
        b = np.cumsum(b)
        t = np.linspace(0, tn, n + 1)
        s = s0 * np.exp(self.mu * t + self.sigma * b)
        return t, s


class MultiGbm():

    def __init__(self, drift=None, Sigma=None):
        """ Multivariate geometric Brownian motion with drift vector and covariance of log-returns
        """
        self.drift = drift
        self.Sigma = Sigma

    def __str__(self):
        return "Drift vector = " + str(self.drift) + "\n Covariance = " + str(self.Sigma)

    def fit(self, X, ema_filter=0.0, timescale=1.0 / 252.0):
        """ Estimate drift vector and covariance matrix of log-returns using either
        a naive sample estimate or exponentially weighted averages.
        """
        # Assuming X is a pandas series of log-returns
        # Fit a GBM model with naive estimators
        if ema_filter == 0.0:
            # print("Using naive-estimates")
            self.Sigma = np.cov(X, rowvar=False) / timescale
            self.drift = np.mean(X, axis=0) / timescale + 0.5 * np.diagonal(self.Sigma)
            self.drift = self.drift.values
        if ema_filter > 0.0:
            # print("Using EMA filter")
            self.Sigma = ewmc(X, ema_filter) / timescale
            self.drift = ema(X, ema_filter) / timescale + 0.5 * np.diagonal(self.Sigma)
            self.Sigma = self.Sigma.values
        elif ema_filter < 0:
            raise ValueError("'ema_filter' must be non-negative")
        return None

    def kelly_criterion(self, r=0):
        """ Compute log-optimal portfolio allocations
        """
        # print(mv_solver(self.drift - r, self.Sigma))
        # print(kc(self.drift-r,self.Sigma))
        return mv_solver(self.drift - r, self.Sigma)
        # return kc(self.drift-r,self.Sigma)

    def min_variance(self):
        """ Compute the minimum variance portfolio
        """
        d = self.Sigma.shape[0]
        return mv_solver(np.zeros(d), self.Sigma)
