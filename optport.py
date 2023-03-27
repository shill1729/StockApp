import numpy as np
import quadprog


def unconstrained_control(drift, covariance):
    # Calculate the inverse of the covariance matrix
    inv_covariance_matrix = np.linalg.inv(covariance)

    # Multiply the inverse covariance matrix by the drift vector
    result = np.dot(inv_covariance_matrix, drift)

    return result


# Wang and Carreira-Perpinan's algorithm for prob. simplex projection.
def project_to_simplex(y):
    # Sort defaults to increasing order
    u = np.sort(y)[::-1]  # Use indices to reverse to obtain decreasing
    d = u.shape[0]
    idx = np.arange(1, d + 1)
    # We need rho to be in I_d := {1,2,...,d}
    rho = np.max(np.where(u + (1 - np.cumsum(u)) / idx > 0)) + 1
    lam = (1 / rho) * (1 - sum(u[:rho]))
    return np.maximum(y + lam, 0)


def kelly_criterion(drift, covariance):
    weights = project_to_simplex(unconstrained_control(drift, covariance))
    return weights


def diffusion_kelly(x, drift, covariance):
    return kelly_criterion(drift(x), covariance(x))


def ito_kelly(t, x, drift, covariance):
    return kelly_criterion(drift(t, x), covariance(t, x))


def path_dependent_kelly(t, x, drift, covariance):
    """ Path dependent optimal control for log-growth.

    :param t vector
    :param x vector
    :param drift function of vector t and (n,d) matrix X
    :param covariance matrix function of vector t and matrix X
    """
    n = t.shape[0]
    if len(x.shape) == 1:
        return kelly_criterion(drift(t[:n], x[:n]), covariance(t[:n], x[:n]))
    else:
        return kelly_criterion(drift(t[:n], x[:n, :]), covariance(t[:n], x[:n, :]))


def mv_solver(drift, Sigma, betas=None, restraint=1.0):
    """ Wrapper to quadprog's QP solver for the following portfolio optimization:
            min 0.5 x^T Sigma x-u^T x
             s.t x_i >= 0 for all i=1,...,d
                1^Tx = 1
                beta^T x = 0

    If the drift parameters are all zero, then we minimize variance, otherwise, we maximize the growth-rate.
    Parameters:
    --------
    drift : array, shape = (d,) for GBM, the vector of drifts of log-returns, for discrete-time the mean of arithmetic returns
    Sigma : array, shape = (d,d) for GBM the covariance matrix of log-returns, for discrete time either the
    covariance matrix of arithmetic returns, or mixed moment matrix E(R^TR)_{ij}=E(R_i R_j)
    betas : array, shape = (d,) for discrete-time the beta coefficients in linear regression against a market index, defaults to None.
    restraint : float, the max value of wealth to invest in

    Returns:
    --------
    optimal_weights : array, shape = (d,) the optimal allocation vector
    optimal_value : float, the optimal value of the objective function.


    """
    d = drift.shape[0]
    # For maximizing growth rate
    objective_function_sign = -1
    # For budget constraints only 1, for budget+beta-neutral use 2,
    num_eq_constr = 1
    # If a zero-vector drift is passed, then we are minimizing variance.
    if all(drift == np.zeros(d)):
        objective_function_sign = 1

    ones = np.ones(d)
    zeros = np.zeros(d)
    # The representation of constraints for long only is:
    # - sum x_i = -restraint (with meq = 1), and x_i >0
    if betas is not None:
        num_eq_constr = 2
        bvec = np.array([restraint, 0])
        bvec = np.hstack((bvec, zeros))
        A = np.hstack((ones[:, None], betas, np.eye(d)))
    else:
        bvec = np.array([restraint])
        bvec = np.hstack((bvec, zeros))
        A = np.hstack((ones[:, None], np.eye(d)))

    # Now pass to solve_qp to solve it!
    w = quadprog.solve_qp(G=Sigma, a=drift, C=A, b=bvec, meq=num_eq_constr)
    optimal_weights = w[0]
    optimal_value = objective_function_sign * w[1]
    return optimal_weights, optimal_value


if __name__ == "__main__":
    drift = np.array([0.025, 0.03])
    Sigma = np.array([[0.5, 0.1], [0.1, 0.2]])
    Sigma = Sigma @ Sigma.T
    weights1 = unconstrained_control(drift, Sigma)
    weights2 = kelly_criterion(drift, Sigma)
    weights3, g = mv_solver(drift, Sigma)
    print("Unconstrained")
    print(weights1)
    print("Projected Unconstrained")
    print(weights2)
    print("Quadratic solver")
    print(weights3)
    print("Growth of unconstrained")
    print(drift.T.dot(weights1)-0.5*weights1.T@Sigma@weights1)
    print("Growth of projected")
    print(drift.T.dot(weights2) - 0.5 * weights2.T @ Sigma @ weights2)
    print("Growth of quadratic")
    print(g)
