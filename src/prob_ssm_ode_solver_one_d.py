from ctypes import Union
from typing import Self, Tuple
import jax
import quadax
import diffrax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array
from functools import partial


jax.scipy.stats.norm.pdf


@partial(jax.jit, static_argnames=["q"])
def discrete_transition_matrix(q: int, h: float) -> Array:
    # TODO pass d, allow for d>1
    # constructs A[i, j] = h^(j - i) / (j - i)!  where j - i >= 0.
    A_exponents = jnp.stack([jnp.arange(q + 1) - i for i in range(q + 1)])
    A = jnp.triu(h ** A_exponents / jax.scipy.special.factorial(A_exponents))
    return A


@partial(jax.jit, static_argnames=["q"])
def discrete_diffusion_matrix(q: int, h: float) -> Array:
    # TODO pass d, allow for d>1
    # Q is a bit more involved, see book "Probabilistic Numerics" (Hennig, Osborne, Kersting)
    # p.51 (chapter 5: Gauss-Markov Processes: Filtering and SDEs) for the formula
    # (a reference for a detailed derivation is provided there as well).
    Q_j = jnp.stack((jnp.arange(q + 1) + 1,) * (q + 1))
    Q_exponent = 2 * q + 3 - Q_j.T - Q_j
    Q_divisor_j = jax.scipy.special.factorial(q + 1 - Q_j)
    Q = (h ** Q_exponent
         / (
             Q_exponent
             * Q_divisor_j
             * Q_divisor_j.T
           ))
    return Q


@partial(jax.jit, static_argnames=["q"])
def discretized_sde(q: int, h: float) -> tuple[Array, Array]:
    # TODO pass d, allow for d>1
    A = discrete_transition_matrix(q, h)
    Q = discrete_diffusion_matrix(q, h)
    return A, Q


@jax.jit
def next_predictive_mean_and_cov(A, m_f, P_f, Q):
    m_p = A @ m_f  # predictive mean
    P_p = A @ P_f @ A.T + Q  # predictive covariance
    return m_p, P_p


def ekf_residual(f, t, m_p, H, H0, f_args, order=1) -> tuple[Array, Array]:
    # Not jit-able for arbitrary f; we would need: jit-able f and then recompile for every f.
    m_p0 = (H0 @ m_p).squeeze()
    if order == 0:
        f_at_m_p = f(t, m_p0, f_args)
        H_hat = H
    elif order == 1:
        f_at_m_p, J_f = jax.value_and_grad(f, argnums=1)(t, m_p0, f_args)
        H_hat = H - J_f * H0  # regular * in 1-d case
    else:
        raise ValueError(f"Invalid value for approximation order of the EKF (has to be 0 or 1): {order}")
    z_hat = f_at_m_p - H @ m_p  # residual
    return z_hat, H_hat


@jax.jit
def local_error_estimate(Q, H_hat) -> Array:
    return jnp.sqrt(H_hat @ Q @ H_hat.T)


@jax.jit
def next_filtering_mean_and_cov(R, m_p, P_p, z_hat, H_hat) -> tuple[Array, Array, Array]:
    # Kalman filter statistics
    S = H_hat @ P_p @ H_hat.T + R  # innovation_covariance
    S_inv = jnp.linalg.inv(S)
    K = P_p @ H_hat.T @ S_inv  # Kalman gain

    # Next filtering mean/cov
    m_f = m_p + K @ z_hat
    P_f = (jnp.eye(P_p.shape[-1]) - K @ H_hat) @ P_p

    # can be used to estimate the scale/variance of the IWP prior later on
    # TODO actually, not sure if we want to use this. Look this up later on and maybe remove it.
    global_uncertainty_term = z_hat.T @ S_inv @ z_hat
    return m_f, P_f, global_uncertainty_term


def ode_filter_step(
        m_f, P_f,
        f, f_args,
        t,
        q,
        h,
        H,
        H0,
        R,
        reltol=1e-3,
        stepsize_safety_factor=0.9,
        stepsize_min_change=0.2,
        stepsize_max_change=10.0,  # as in https://arxiv.org/abs/2012.08202
        EKF_approximation_order=1
        ) -> Tuple[Array, Array, Array, Array, Array]:

    # TODO this sucks, refactor, do full step here and just report next stepsize + local error,
    # then let caller decide if step is rejected or not.
    # Also, jit with static args f, q and order

    while True:

        A, Q_hat = discretized_sde(q, h)  # A: transition matrix, B: diffusion matrix
        m_p, P_p = next_predictive_mean_and_cov(A, m_f, P_f, Q_hat)

        z_hat, H_hat = ekf_residual(f, t, m_p, H, H0, f_args, order=EKF_approximation_order)

        sigma_hat = z_hat.T @ z_hat / (H @ Q_hat @ H.T)

        Q = Q_hat * sigma_hat ** 2
        local_error = local_error_estimate(Q, H_hat)
        stepsize_factor = stepsize_safety_factor * (reltol / local_error) ** (1.0 / q + 1.0)
        stepsize_factor = jax.lax.clamp(stepsize_min_change, stepsize_factor, stepsize_max_change)
        h *= stepsize_factor
        if local_error <= reltol:
            # accept step
            break
    # TODO variable stepsize: check error (and reject if too large) and calculate next h and t.
    P_p = P_p - Q_hat + Q

    m_f_next, P_f_next, global_uncertainty_term = \
            next_filtering_mean_and_cov(R, m_p, P_p, z_hat, H_hat)
    return m_f_next, P_f_next, m_p, P_p, h.squeeze()



def ode_ssm_smoother_update(
        m_f, P_f,
        A,
        m_p_next, P_p_next,
        m_s_next, P_s_next) -> Tuple[Array, Array]:
    G = P_f @ A.T @ jnp.linalg.inv(P_p_next)  # gain
    m_s = m_f + G @ (m_s_next - m_p_next)  # posterior mean
    P_s = P_f + G @ (P_s_next - P_p_next) @ G.T  # posterior covariance
    return m_s, P_s


def ode_smoother(
        x_initial, P0,
        f, f_args,
        t0,
        t1,
        q,
        h,
        R,
        adaptive_stepsize=False,
        apply_smoother=True,
        approximation_order=1):

    # Assume P0 == 0 (initial x is known without uncertainty)
    P0 = jnp.zeros((q + 1, q + 1))

    # TODO change N to t1 and for to while loop to vary num of steps

    # TODO for constant stepsize h: implement p.50; footnote 16 for better efficiency/stability
    d = x_initial.shape[-1]  # TODO how to handle d>1? See Kersting Sullivan Hennig 2020 Appendix B
    assert d == 1  # TODO

    # TODO where in book was this? also this so far only works for q=1
    # TODO p.290, I'm not sure whether this is correct yet even for q=1, look into this later.
    x0 = jnp.stack([x_initial, f(t0, x_initial, f_args)])

    if not isinstance(t0, Array):
        t0 = jnp.array(t0)
    ts = [t0]
    predictive_params = [(x0, P0)]  # This should be initialized empty, putting something in here to keep it in line with indexing in the book
    filtering_params = [(x0, P0)]

    H0 = jnp.zeros((1, q + 1)).at[0, 0].set(1)
    H = jnp.zeros((1, q + 1)).at[0, 1].set(1)  # TODO d > 1

    while True:
        t = ts[-1]
        m_f, P_f = filtering_params[-1]
        m_f_next, P_f_next, m_p, P_p, next_h = \
                ode_filter_step(m_f, P_f, f, f_args, t, q, h, H, H0, R,
                                EKF_approximation_order=approximation_order)
        filtering_params.append((m_f_next, P_f_next))
        predictive_params.append((m_p, P_p))
        if adaptive_stepsize:
            h = next_h
        ts.append(t + h)
        if t >= t1:
            break

    if apply_smoother:
        smoothing_params = [filtering_params[-1]]
        for n in range(len(filtering_params) - 2, -1, -1):  # shift by 1 to keep in line with Algorithm 5.4
            m_s_next, P_s_next = smoothing_params[0]  # This also yields m_s(n + 1). We just build it from the front
            m_p_next, P_p_next = predictive_params[n + 1]
            h = ts[n + 1] - ts[n]
            A = discrete_transition_matrix(q, h)
            m_f, P_f = filtering_params[n]
            m_s, P_s = ode_ssm_smoother_update(m_f, P_f, A, m_p_next, P_p_next, m_s_next, P_s_next)
            smoothing_params.insert(0, (m_s, P_s))
    else:
        smoothing_params = None

    # Predictive params are not actually useful,
    # return them anyways in case they are of interest eg. for visualizations
    return ts, filtering_params, smoothing_params, predictive_params




def linear_vf(t, x, args):
    alpha = args[0]
    return alpha * x




if __name__ == "__main__":
    x0 = jnp.array([0.4])
    P0 = 0
    alpha = 0.5
    f = linear_vf
    f_args = (alpha, )
    q = 1
    t0 = 0
    initial_stepsize = 5e-3
    N = 500
    t1 = t0 + initial_stepsize * N
    R = 0
    approximation_order = 1
    ts, fp, sp, pp = ode_smoother(x0, P0, f, f_args, t0, t1, q, initial_stepsize, R, approximation_order=approximation_order)

    term = diffrax.ODETerm(f)
    solver = diffrax.Tsit5()
    dt0 = 0.1
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, N+1))
    sol = diffrax.diffeqsolve(term, solver, t0, t1, dt0, x0, args=f_args, saveat=saveat)

    plt.plot(sol.ts, sol.ys, label="Tsit5")

    filtering_xs = list(map(lambda el: el[0][0], fp))
    predictive_xs = list(map(lambda el: el[0][0], pp))
    smoothing_xs = list(map(lambda el: el[0][0], sp))
    plt.plot(ts, filtering_xs, label="Filtering")
    plt.plot(ts, smoothing_xs, label="Smoothing")
    plt.plot(ts, predictive_xs, label="Predictive")
    plt.legend()
    plt.show()
