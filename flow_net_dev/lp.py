import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


def make_lp_layer(n: int, m: int):
    x = cp.Variable(n, nonneg=True)  # x ≥ 0
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    c = cp.Parameter(n)

    prob = cp.Problem(cp.Minimize(c @ x), [A @ x == b])
    assert prob.is_dpp()  # disciplined parametrised program

    return CvxpyLayer(prob, parameters=[A, b, c], variables=[x])
