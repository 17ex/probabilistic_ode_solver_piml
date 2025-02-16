from ctypes import Union
from typing import Self, Tuple, Optional
import jax
import quadax
import diffrax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array
from functools import partial

# TODO decide if using fp64 is a good idea
jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnames=["d", "q"])
def discrete_transition_matrix(d: int, q: int, h: float) -> Array:
    # constructs A[i, j] = h^(j - i) / (j - i)!  where j - i >= 0.
    A_exponents = jnp.stack([jnp.arange(q + 1) - i for i in range(q + 1)])
    A = jnp.triu(h ** A_exponents / jax.scipy.special.factorial(A_exponents))
    return jnp.kron(A, jnp.eye(d))


@partial(jax.jit, static_argnames=["d", "q"])
def discrete_diffusion_matrix(d: int, q: int, h: float) -> Array:
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
    return jnp.kron(Q, jnp.eye(d))


@partial(jax.jit, static_argnames=["d", "q"])
def discretized_sde(d: int, q: int, h: float) -> tuple[Array, Array]:
    A = discrete_transition_matrix(d, q, h)
    Q = discrete_diffusion_matrix(d, q, h)
    return A, Q


@partial(jax.jit, static_argnames=["d", "q"])
def ssm_projection_matrices(d, q) -> tuple[Array, Array]:
    # Projections: project SSM state -> y (H0) or SSM state -> y' (H)
    H0 = jnp.zeros((1, q + 1)).at[0, 0].set(1.0)
    H = jnp.zeros((1, q + 1)).at[0, 1].set(1.0)
    Id = jnp.eye(d)
    H0 = jnp.kron(H0, Id)
    H = jnp.kron(H, Id)
    return H0, H


@jax.jit
def next_predictive_mean_and_cov(A, m_f, P_f, Q):
    m_p = A @ m_f  # predictive mean
    P_p = A @ P_f @ A.T + Q  # predictive covariance
    return m_p, P_p


@partial(jax.jit, static_argnames=["f", "f_args", "approximation_order"])
def ekf_residual(f, t, m_p, H, H0, f_args, approximation_order) -> tuple[Array, Array]:
    m_p0 = H0 @ m_p
    f_at_m_p = f(t, m_p0, f_args)
    if approximation_order == 0:
        H_hat = H
    elif approximation_order == 1:
        J_f = jax.jacfwd(f, argnums=1)(t, m_p0, f_args)
        H_hat = H - J_f @ H0  # regular * in 1-d case
    else:
        raise ValueError(f"Invalid value for approximation order of the EKF (has to be 0 or 1): {approximation_order}")
    z_hat = f_at_m_p - H @ m_p  # residual
    return z_hat, H_hat


@jax.jit
def local_error_estimate(Q, H_hat) -> Array:
    # Max over dimensions d; It's not entirely clear to me if this is the
    # correct choice, but this seems to be a reasonable choice.
    return jnp.sqrt((H_hat @ Q @ H_hat.T).max())


@jax.jit
def next_filtering_mean_and_cov(R, m_p, P_p, z_hat, H_hat) -> tuple[Array, Array]:
    # Kalman filter statistics
    S = H_hat @ P_p @ H_hat.T + R  # innovation_covariance
    K = P_p @ H_hat.T @ jnp.linalg.inv(S)  # Kalman gain
    # Next filtering mean/cov
    m_f = m_p + K @ z_hat
    P_f = (jnp.eye(H_hat.shape[1]) - K @ H_hat) @ P_p
    return m_f, P_f


@partial(jax.jit, static_argnames=["f", "f_args", "d", "q", "adaptive_stepsize", "EKF_approximation_order"])
def ode_filter_step(
        m_f, P_f,
        f, f_args,
        t,
        d, q,
        h,
        H,
        H0,
        R,
        reltol,
        adaptive_stepsize,
        stepsize_safety_factor,
        stepsize_min_change,
        stepsize_max_change,  # as in https://arxiv.org/abs/2012.08202
        EKF_approximation_order
        ) -> Tuple[Array, Array, Array, Array, Array, Array]:
    A, Q_hat = discretized_sde(d, q, h)  # A: transition matrix, Q: diffusion matrix
    m_p, P_p = next_predictive_mean_and_cov(A, m_f, P_f, Q_hat)

    z_hat, H_hat = ekf_residual(f, t, m_p, H, H0, f_args,
                                approximation_order=EKF_approximation_order)
    sigma_hat = z_hat.T @ jnp.linalg.inv(H @ Q_hat @ H.T) @ z_hat

    Q = Q_hat * sigma_hat
    P_p = P_p - Q_hat + Q  # Use Q for P_p instead of Q_hat (as P_p = ... + Q_hat)

    local_error = local_error_estimate(Q, H_hat)

    m_f_next, P_f_next = next_filtering_mean_and_cov(R, m_p, P_p, z_hat, H_hat)

    if adaptive_stepsize:
        stepsize_factor = stepsize_safety_factor * (reltol / local_error) ** (1.0 / (q + 1.0))
        stepsize_factor = jax.lax.clamp(stepsize_min_change, stepsize_factor, stepsize_max_change)
        h = (h * stepsize_factor).squeeze()

    return m_f_next, P_f_next, m_p, P_p, local_error, h


@jax.jit
def ode_ssm_smoother_update(
        m_f, P_f,
        A,
        m_p_next, P_p_next,
        m_s_next, P_s_next) -> Tuple[Array, Array]:
    G = P_f @ A.T @ jnp.linalg.inv(P_p_next)  # gain
    m_s = m_f + G @ (m_s_next - m_p_next)  # posterior mean
    P_s = P_f + G @ (P_s_next - P_p_next) @ G.T  # posterior covariance
    return m_s, P_s


@jax.jit
def linear_interpolation_mean_cov(t0, t1, t, m0, m1, P0, P1) -> Tuple[Array, Array]:
    w0 = (t - t0) / (t1 - t0)
    w1 = (t1 - t) / (t1 - t0)
    m = w0 * m0 + w1 * m1
    P = w0 * P0 + w1 * P1
    return m, P


def ode_filter(
        y_initial, P0,
        f, f_args,
        t0,
        t1,
        q,
        h,
        R,
        saveat: Optional[Array] = None,
        reltol=1e-1,
        adaptive_stepsize=True,
        stepsize_safety_factor=0.9,
        stepsize_min_change=0.2,
        stepsize_max_change=10.0,  # as in https://arxiv.org/abs/2012.08202
        approximation_order=1) -> dict[str, Array]:

    if saveat is None:
        save_all = True
    else:
        save_all = False

    # Ensure these are arrays
    h = jnp.array(h)
    t0 = jnp.array(t0)
    reltol = jnp.array(reltol)
    stepsize_safety_factor = jnp.array(stepsize_safety_factor)
    stepsize_min_change = jnp.array(stepsize_min_change)
    stepsize_max_change = jnp.array(stepsize_max_change)

    # TODO list: (priority ordered)
    # TODO Nordsieck-like rescaling
    # TODO Square-root Kalman filter
    # TODO support for q>1
    d = y_initial.shape[-1]
    assert q == 1

    H0, H = ssm_projection_matrices(d, q)
    # TODO Probabilistic Numerics book, p.290, this here just works for q=1
    # Initialization
    x0 = jnp.stack([y_initial, f(t0, y_initial, f_args)]).flatten()
    P0 = jnp.kron(jnp.zeros((q + 1, q + 1)), jnp.eye(d))  # Assume no uncertainty in x0
    # Init. book-keeping of results
    rejected_steps = 0
    ts: list[Array] = [t0]
    filtering_means: list[Array] = [x0]
    filtering_covs: list[Array] = [P0]
    predictive_means: list[Array] = [x0]  # Only used further if save_all
    predictive_covs: list[Array] = [P0]
    # Init. loop variables
    t_prev = t0
    m_f_prev = x0  # Filtering mean (of previous step)
    P_f_prev = P0  # Filtering cov (of previous step)
    save_t_ind = 0

    def store_vals(t, m_f, P_f, m_p, P_p):
        ts.append(t)
        filtering_means.append(m_f)
        filtering_covs.append(P_f)
        if save_all:
            predictive_means.append(m_p)
            predictive_covs.append(P_p)

    # The ODE filter/solver loop
    while t_prev < t1:
        m_f, P_f, m_p, P_p, local_error, next_h = \
                ode_filter_step(m_f_prev, P_f_prev, f, f_args, t_prev, d, q, h, H, H0, R,
                                reltol=reltol,
                                adaptive_stepsize=adaptive_stepsize,
                                stepsize_safety_factor=stepsize_safety_factor,
                                stepsize_min_change=stepsize_min_change,
                                stepsize_max_change=stepsize_max_change,
                                EKF_approximation_order=approximation_order)
        t = t_prev + h
        if adaptive_stepsize:
            h = next_h
            if local_error > reltol:
                # Reject current step (too inaccurate)
                rejected_steps += 1
                continue
        # Step is accepted: store results
        if save_all:
            store_vals(t, m_f, P_f, m_p, P_p)
        else:
            while save_t_ind < saveat.shape[0] and saveat[save_t_ind] <= t:
                # interpolate over saveat points
                save_t = saveat[save_t_ind]
                m, P = linear_interpolation_mean_cov(
                        t_prev, t, save_t, m_f_prev, m_f, P_f_prev, P_f)
                store_vals(save_t, m, P, None, None)
                save_t_ind += 1
            if save_t_ind == saveat.shape[0]:
                break  # May as well stop here.
        # Update loop variables
        t_prev, m_f_prev, P_f_prev = t, m_f, P_f

    # TODO consider adding a class for solver results.
    # Format and return results
    result_dict = {"num_rejected": jnp.array(rejected_steps)}
    if save_all:
        result_dict["ys_predictive"] = jnp.array(predictive_means).squeeze()
        result_dict["covs_predictive"] = jnp.array(predictive_covs).squeeze()
    else:
        # Remove first element (as it was inserted automatically)
        # to return elements only at saveat.
        for l in [ts, filtering_means, filtering_covs]:
            l.pop(0)
    result_dict["ts"] = jnp.array(ts).squeeze()
    result_dict["ys"] = jnp.array(filtering_means).squeeze()
    result_dict["covs"] = jnp.array(filtering_covs).squeeze()
    result_dict["H"] = H
    result_dict["H0"] = H0
    result_dict["d"] = jnp.array(d)
    result_dict["q"] = jnp.array(q)
    return result_dict


def ode_smoother(filter_results: dict[str, Array],
                 saveat: Optional[Array] = None) -> dict[str, Array]:
    # Setup, retrieve parameters
    predictive_means = filter_results["ys_predictive"]
    predictive_covs = filter_results["covs_predictive"]
    filtering_means = filter_results["ys"]
    filtering_covs = filter_results["covs"]
    ts = filter_results["ts"]
    N, d, q = ts.shape[0], filter_results["d"].item(), filter_results["q"].item()
    if saveat is not None:
        save_all = False
        save_t_ind = saveat.shape[0] - 1
    else:
        save_all = True
        save_t_ind = 0
    t_next = ts[-1]
    m_s_next = filtering_means[-1, ...]
    P_s_next = filtering_covs[-1, ...]
    smoothing_ts = [t_next]
    smoothing_means = [m_s_next]
    smoothing_covs = [P_s_next]

    def store_vals(t, m, P):
        smoothing_ts.insert(0, t)
        smoothing_means.insert(0, m)
        smoothing_covs.insert(0, P)

    # The actual smoothing iterations
    for n in range(N - 2, -1, -1):  # shift by 1 due to zero-based indexing
        m_p_next, P_p_next = predictive_means[n + 1, ...], predictive_covs[n + 1, ...]
        m_f, P_f = filtering_means[n, ...], filtering_covs[n, ...]
        t = ts[n]
        h = t_next - t
        A = discrete_transition_matrix(d, q, h)
        # TODO h and A can go in the step function (and rename below to _step)
        m_s, P_s = ode_ssm_smoother_update(m_f, P_f, A, m_p_next, P_p_next, m_s_next, P_s_next)

        # Store results
        if save_all:
            store_vals(t, m_s, P_s)
        else:
            while save_t_ind >= 0 and saveat[save_t_ind] >= t:
                # interpolate over saveat points
                save_t = saveat[save_t_ind]
                m, P = linear_interpolation_mean_cov(
                        t, t_next, save_t, m_s, m_s_next, P_s, P_s_next)
                store_vals(save_t, m, P)
                save_t_ind -= 1
            if save_t_ind < 0:
                break  # May as well stop here.
        m_s_next, P_s_next, t_next = m_s, P_s, t

    del filter_results["ys_predictive"]
    del filter_results["covs_predictive"]
    if not save_all:
        for l in [smoothing_ts, smoothing_means, smoothing_covs]:
            l.pop()  # Remove first inserted item (which was not inserted according to saveat).
    filter_results["ts"] = jnp.array(smoothing_ts).squeeze()
    filter_results["ys"] = jnp.array(smoothing_means).squeeze()
    filter_results["covs"] = jnp.array(smoothing_covs).squeeze()
    return filter_results


def ode_filter_and_smoother(
        y_initial, P0,
        f, f_args,
        t0,
        t1,
        q,
        h,
        R,
        saveat: Optional[Array] = None,
        reltol=1e-1,
        adaptive_stepsize=True,
        stepsize_safety_factor=0.9,
        stepsize_min_change=0.2,
        stepsize_max_change=10.0,  # as in https://arxiv.org/abs/2012.08202
        approximation_order=1,
        apply_smoother=True
        ) -> dict[str, Array]:
    results = ode_filter(
        y_initial, P0,
        f, f_args,
        t0,
        t1,
        q,
        h,
        R,
        saveat=(None if apply_smoother else saveat),
        reltol=reltol,
        adaptive_stepsize=adaptive_stepsize,
        stepsize_safety_factor=stepsize_safety_factor,
        stepsize_min_change=stepsize_min_change,
        stepsize_max_change=stepsize_max_change,
        approximation_order=approximation_order
    )
    if apply_smoother:
        results = ode_smoother(results, saveat=saveat)
    return results


@partial(jax.jit, static_argnames=["args"])
def linear_vf(t, y, args):
    alpha = args[0]
    return alpha * y



if __name__ == "__main__":
    y0 = jnp.array([4.0, 3.5])
    P0 = 0
    alpha = 2.0
    f = linear_vf
    f_args = (alpha, )
    q = 1
    t0 = 0.0
    t1 = 3.0
    initial_stepsize = 5e-3
    adaptive_stepsize = False
    N = 100
    R = 0  # ~measurement cov (-> cov of z)
    approximation_order = 1
    apply_smoother = True
    grid = jnp.linspace(t0, t1, N)
    prob_sol = ode_filter_and_smoother(y0, P0, f, f_args, t0, t1, q, initial_stepsize, R,
                                       adaptive_stepsize=adaptive_stepsize,
                                       approximation_order=approximation_order,
                                       saveat=grid,
                                       apply_smoother=apply_smoother)

    term = diffrax.ODETerm(f)
    solver = diffrax.Tsit5()
    dt0 = 0.1
    saveat = diffrax.SaveAt(ts=grid)
    sol = diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0,
                              args=f_args,
                              saveat=saveat)
    true_d = f(sol.ts, sol.ys, f_args)

    label_prefix = "Smoothing" if apply_smoother else "Filtering"
    ts = prob_sol["ts"]
    # TODO  in the solvers, dont store x but only y instead.
    ys = prob_sol["ys"] @ prob_sol["H0"].T
    y_stddevs = jnp.sqrt(jnp.diagonal(prob_sol["H0"] @ prob_sol["covs"] @ prob_sol["H0"].T, axis1=1, axis2=2))

    # TODO good plotting
    plt.plot(sol.ts, sol.ys, label="Tsit5", linewidth=4, color="black")
    plt.plot(ts, ys, label=f"{label_prefix} mean", linewidth=2, color="green")
    plt.fill_between(ts, ys[:, 0] - y_stddevs[:, 0], ys[:, 0] + y_stddevs[:, 0], label=f"{label_prefix} cov", linewidth=1, color="green", alpha=0.3)
    plt.fill_between(ts, ys[:, 1] - y_stddevs[:, 1], ys[:, 1] + y_stddevs[:, 1], label=f"{label_prefix} cov", linewidth=1, color="green", alpha=0.3)
    plt.legend()
    plt.show()
