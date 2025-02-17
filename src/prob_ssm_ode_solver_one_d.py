from typing import Tuple, Optional
import jax
import scipy
import numpy as np
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt
from jaxtyping import Array
from functools import partial

# Use float64 precision
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


def discrete_transition_matrix_ns(d: int, q: int) -> Array:
    # constructs A[i, j] = binom(q - i, q - j) if j - i >= 0, else 0
    # calculation in int should (?) be more accurate than in float (with gamma)
    K = q - np.repeat(np.expand_dims(np.arange(q + 1), 0), q + 1, 0)
    A = jnp.triu(scipy.special.comb(K.T, K)).astype(float)  # triu not needed, left for clarity
    return jnp.kron(A, jnp.eye(d))


def discrete_diffusion_matrix_ns(d: int, q: int) -> Array:
    # Q is a bit more involved, see book "Probabilistic Numerics" (Hennig, Osborne, Kersting)
    # p.51 (chapter 5: Gauss-Markov Processes: Filtering and SDEs) for the formula
    # (a reference for a detailed derivation is provided there as well).
    Q_j = jnp.stack((jnp.arange(q + 1) + 1,) * (q + 1))
    Q = 1.0 / (2 * q + 3 - Q_j.T - Q_j)
    return jnp.kron(Q, jnp.eye(d))


def discretized_sde_ns(d: int, q: int) -> tuple[Array, Array]:
    A = discrete_transition_matrix_ns(d, q)
    Q = discrete_diffusion_matrix_ns(d, q)
    return A, Q


def ssm_projection_matrices(d, q) -> tuple[Array, Array]:
    # Projections: project SSM state -> y (H0) or SSM state -> y' (H)
    H0 = jnp.zeros((1, q + 1)).at[0, 0].set(1.0)
    H = jnp.zeros((1, q + 1)).at[0, 1].set(1.0)
    Id = jnp.eye(d)
    H0 = jnp.kron(H0, Id)
    H = jnp.kron(H, Id)
    return H0, H


@partial(jax.jit, static_argnames=["d", "q"])
def nordsieck_coord_transformation(d, q, h) -> tuple[Array, Array]:
    qs = jnp.flip(jnp.arange(q + 1), 0)
    # cumprod vs factorial? Probably will make no difference as both
    # are constant under jit, but maybe test this out later?
    T_diag = jnp.sqrt(h) * (h ** qs / jax.scipy.special.factorial(qs))
    T_inv_diag = 1.0 / T_diag
    T = jnp.kron(jnp.diag(T_diag), jnp.eye(d))
    T_inv = jnp.kron(jnp.diag(T_inv_diag), jnp.eye(d))
    return T, T_inv


@jax.jit
def cho_cov_to_stddev(L, H0) -> Array:
    return jnp.sqrt(jnp.diag(H0 @ L @ L.T @ H0.T))


@jax.jit
def predictive_cov(A, L_f, L_Q):
    C_p_pre = jnp.vstack([(A @ L_f).T, L_Q.T])
    L_p = jax.scipy.linalg.qr(C_p_pre, mode="economic")[1].T
    return L_p


@partial(jax.jit, static_argnames=["f", "f_args", "approximation_order"])
def ekf_residual(f, t, m_p, H, H0, T, f_args, approximation_order) -> tuple[Array, Array]:
    m_p0 = H0 @ T @ m_p
    f_at_m_p = f(t, m_p0, f_args)
    if approximation_order == 0:
        H_hat = H
    elif approximation_order == 1:
        J_f = jax.jacfwd(f, argnums=1)(t, m_p0, f_args)
        H_hat = H - J_f @ H0  # regular * in 1-d case
    else:
        raise ValueError(f"Invalid value for approximation order of the EKF (has to be 0 or 1): {approximation_order}")
    z_hat = f_at_m_p - H @ T @ m_p  # residual
    H_hat = H_hat @ T
    return z_hat, H_hat


@jax.jit
def local_error_estimate(L_Q, H_hat) -> Array:
    # Max over dimensions d; It's not entirely clear to me if this is the
    # correct choice, but this seems to be a reasonable choice.
    return jnp.sqrt((H_hat @ L_Q @ L_Q.T @ H_hat.T).max())


@jax.jit
def next_filtering_mean_and_cov(m_p, L_p, z_hat, H_hat) -> tuple[Array, Array]:
    # Kalman filter statistics
    S_pre = (H_hat @ L_p).T
    L_S = jax.scipy.linalg.qr(S_pre, mode="economic")[1].T  # innovation cov
    C_cross = L_p @ L_p.T @ H_hat.T
    L_S_inv = jax.scipy.linalg.solve_triangular(L_S, jnp.eye(L_S.shape[0]), lower=True)
    K = C_cross @ L_S_inv.T @ L_S_inv  # Kalman gain
    # Next filtering mean/cov
    # note: as compared to https://arxiv.org/pdf/2012.10106,
    # our sign of z is different (due to different definition).
    m_f = m_p + K @ z_hat  # + or plus here?
    L_f = (jnp.eye(H_hat.shape[1]) - K @ H_hat) @ L_p
    return m_f, L_f


@partial(jax.jit, static_argnames=["f", "f_args", "d", "q", "adaptive_stepsize", "EKF_approximation_order"])
def ode_filter_step(
        m_f, L_f,
        f, f_args,
        t, h,
        d, q,
        A, L_Q_hat,
        H0, H,
        reltol,
        adaptive_stepsize,
        stepsize_safety_factor,
        stepsize_min_change,
        stepsize_max_change,  # as in https://arxiv.org/abs/2012.08202
        EKF_approximation_order
        ) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    T, T_inv = nordsieck_coord_transformation(d, q, h)
    m_f = T_inv @ m_f
    L_f = T_inv @ L_f

    m_p = A @ m_f  # predictive mean
    z_hat, H_hat = ekf_residual(f, t, m_p, H, H0, T, f_args,
                                approximation_order=EKF_approximation_order)
    # TODO can this be improved?
    # TODO The whole term inside can be precomputed; using triangular_solve
    sigma_hat = z_hat.T @ jnp.linalg.inv(H @ L_Q_hat @ L_Q_hat.T @ H.T) @ z_hat
    L_Q = jnp.sqrt(sigma_hat) * L_Q_hat

    L_p = predictive_cov(A, L_f, L_Q)
    local_error = local_error_estimate(L_Q, H_hat)

    m_f_next, L_f_next = next_filtering_mean_and_cov(m_p, L_p, z_hat, H_hat)
    m_f_next = T @ m_f_next  # Output in original coordinates
    L_f_next = T @ L_f_next
    y = H0 @ m_f_next
    stddev = cho_cov_to_stddev(L_f_next, H0)

    if adaptive_stepsize:
        stepsize_factor = stepsize_safety_factor * (reltol / local_error) ** (1.0 / (q + 1.0))
        stepsize_factor = jax.lax.clamp(stepsize_min_change, stepsize_factor, stepsize_max_change)
        h = (h * stepsize_factor).squeeze()

    return m_f_next, L_f_next, y, stddev, local_error, h, m_p, L_p


@partial(jax.jit, static_argnames=["d", "q"])
def ode_ssm_smoother_update(
        m_f, L_f,
        m_p_next, L_p_next,
        m_s_next, L_s_next,
        A, L_Q, H0, h, d, q) -> Tuple[Array, Array, Array, Array]:
    T, T_inv = nordsieck_coord_transformation(d, q, h)
    m_s_next = T_inv @ m_s_next
    L_s_next = T_inv @ L_s_next
    m_f = T_inv @ m_f
    L_f = T_inv @ L_f
    L_p_inv = jax.scipy.linalg.solve_triangular(
            L_p_next, jnp.eye(L_p_next.shape[0]), lower=True)
    G = L_f @ (A @ L_f).T @ L_p_inv.T @ L_p_inv  # gain
    m_s = m_f + G @ (m_s_next - m_p_next)  # posterior mean
    L_s_pre = jnp.vstack([
        (jnp.eye(A.shape[0]) - G @ A) @ L_f,  # TODO is last one really L_f?
        G @ L_Q,
        G @ L_s_next])
    L_s = jax.scipy.linalg.qr(L_s_pre, mode="economic")[1].T # posterior cov
    m_s = T @ m_s
    L_s = T @ L_s
    y_s = H0 @ m_s
    stddev_s = cho_cov_to_stddev(L_s, H0)
    return m_s, L_s, y_s, stddev_s


@jax.jit
def linear_interpolation_mean_cov(t0, t1, t, m0, m1, P0, P1) -> Tuple[Array, Array]:
    w0 = (t1 - t) / (t1 - t0)
    w1 = (t - t0) / (t1 - t0)
    m = w0 * m0 + w1 * m1
    P = w0 * P0 + w1 * P1
    return m, P


def ode_filter(
        y_initial,
        f, f_args,
        t0,
        t1,
        q,
        h,
        saveat,
        save_for_smoother,
        reltol,
        adaptive_stepsize,
        stepsize_safety_factor,
        stepsize_min_change,
        stepsize_max_change,
        approximation_order) -> dict:
    # Ensure these are arrays
    h = jnp.array(h)
    t0 = jnp.array(t0)
    reltol = jnp.array(reltol)
    stepsize_safety_factor = jnp.array(stepsize_safety_factor)
    stepsize_min_change = jnp.array(stepsize_min_change)
    stepsize_max_change = jnp.array(stepsize_max_change)

    # TODO list: (priority ordered)
    # TODO Square-root Kalman filter
    # TODO support for q>1 (taylor-mode AD for init of x0)
    d = y_initial.shape[-1]
    assert q == 1

    # Initialization
    x0 = jnp.stack([y_initial, f(t0, y_initial, f_args)]).flatten()
    stddev0 = jnp.zeros_like(y_initial)
    L_P0 = jnp.diag(jnp.repeat(stddev0, q + 1))

    A, Q = discretized_sde_ns(d, q)
    H0, H = ssm_projection_matrices(d, q)
    L_Q = jnp.linalg.cholesky(Q)

    # Init. book-keeping of results
    rejected_steps = 0
    ts, ys, stddevs = [], [], []
    if saveat is None:
        ts.append(t0)
        ys.append(y_initial)
        stddevs.append(stddev0)
    filtering_means, filtering_covs, predictive_means, predictive_covs = [], [], [], []
    if save_for_smoother:
        filtering_means.append(x0)
        filtering_covs.append(L_P0)
    # Init. loop variables
    t_prev = t0
    m_f_prev, L_f_prev = x0, L_P0  # Filtering parameters (of previous step)
    y_prev, stddev_prev = y_initial, stddev0
    save_t_ind = 0

    # The ODE filter/solver loop
    while t_prev < t1:
        m_f, L_f, y, stddev, local_error, next_h, m_p, L_p = \
                ode_filter_step(m_f_prev, L_f_prev, f, f_args, t_prev, h, d, q,
                                A, L_Q, H0, H,
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
        if save_for_smoother:
            ts.append(t)
            filtering_means.append(m_f)
            filtering_covs.append(L_f)
            predictive_means.append(m_p)
            predictive_covs.append(L_p)
        else:
            if saveat is None:
                ts.append(t)
                ys.append(y)
                stddevs.append(stddev)
            else:
                while save_t_ind < saveat.shape[0] and saveat[save_t_ind] <= t:
                    # interpolate over saveat points
                    save_t = saveat[save_t_ind]
                    y_i, stddev_i = linear_interpolation_mean_cov(
                            t_prev, t, save_t, y_prev, y, stddev_prev, stddev)
                    ts.append(save_t)
                    ys.append(y_i)
                    stddevs.append(stddev_i)
                    save_t_ind += 1
                if save_t_ind == saveat.shape[0]:
                    break  # May as well stop here.
        # Update loop variables
        t_prev, m_f_prev, L_f_prev = t, m_f, L_f
        y_prev, stddev_prev = y, stddev

    # Format and return results
    if save_for_smoother:
        return {
                "ts": ts,
                "predictive_means": predictive_means,
                "predictive_covs": predictive_covs,
                "filtering_means": filtering_means,
                "filtering_covs": filtering_covs,
                "d": d, "q": q,
                "last_y": y, "last_stddev": stddev,
                "A": A, "L_Q": L_Q, "H0": H0}
    else:
        return {
                "ts": jnp.asarray(ts).squeeze(),
                "ys": jnp.asarray(ys).squeeze(),
                "stddevs": jnp.asarray(stddevs).squeeze(),
                }


def ode_smoother(f_out: dict[str, Array],
                 saveat: Optional[Array] = None) -> dict[str, Array]:
    # Setup, retrieve parameters
    predictive_means = f_out["predictive_means"]
    predictive_covs = f_out["predictive_covs"]
    filtering_means = f_out["filtering_means"]
    filtering_covs = f_out["filtering_covs"]
    ts = f_out["ts"]
    N, d, q = len(ts), f_out["d"], f_out["q"]
    A, L_Q, H0 = f_out["A"], f_out["L_Q"], f_out["H0"]
    save_t_ind = 0
    if saveat is not None:
        save_all = False
        save_t_ind = saveat.shape[0] - 1
    else:
        save_all = True
    t_next = ts[-1]
    m_s_next = filtering_means[-1]
    L_s_next = filtering_covs[-1]
    y_next = f_out["last_y"]
    stddev_next = f_out["last_stddev"]
    smoothing_ts = [t_next]
    smoothing_ys = [y_next]
    smoothing_stddevs = [stddev_next]

    def store_vals(t, y, stddev):
        smoothing_ts.insert(0, t)
        smoothing_ys.insert(0, y)
        smoothing_stddevs.insert(0, stddev)

    # The actual smoothing iterations
    for n in range(N - 2, -1, -1):  # shift by 1 due to zero-based indexing
        # [N - 2, 1]
        # predictive params hold one less element at the start,
        # so eg predictive_means[n] actually corresponds to m_p[n+1]
        m_p_next, L_p_next = predictive_means[n], predictive_covs[n]
        m_f, L_f = filtering_means[n], filtering_covs[n]
        t = ts[n]
        h = ts[n + 1] - t
        m_s, L_s, y, stddev = ode_ssm_smoother_update(
                m_f, L_f, m_p_next, L_p_next, m_s_next, L_s_next, A, L_Q, H0, h, d, q)
        # Store results
        if save_all:
            store_vals(t, y, stddev)
        else:
            while save_t_ind >= 0 and saveat[save_t_ind] >= t:
                # interpolate over saveat points
                save_t = saveat[save_t_ind]
                m, L = linear_interpolation_mean_cov(
                        t, t_next, save_t, y, y_next, stddev, stddev_next)
                store_vals(save_t, m, L)
                save_t_ind -= 1
            if save_t_ind < 0:
                break  # May as well stop here.
        m_s_next, L_s_next, t_next = m_s, L_s, t
        y_next, stddev_next = y, stddev

    if not save_all:
        for l in [smoothing_ts, smoothing_ys, smoothing_stddevs]:
            l.pop()  # Remove first inserted item (which was not inserted according to saveat).
    return {
            "ts": jnp.array(smoothing_ts).squeeze(),
            "ys": jnp.array(smoothing_ys).squeeze(),
            "stddevs": jnp.array(smoothing_stddevs).squeeze()
            }


def ode_filter_and_smoother(
        y_initial,
        f, f_args,
        t0,
        t1,
        q,
        h,
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
        y_initial,
        f, f_args,
        t0,
        t1,
        q,
        h,
        save_for_smoother=apply_smoother,
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
    alpha = 2.0
    f = linear_vf
    f_args = (alpha, )
    q = 1
    t0 = 0.0
    t1 = 3.0
    initial_stepsize = 5e-3
    adaptive_stepsize = False
    N = 100
    approximation_order = 1
    apply_smoother = True
    grid = jnp.linspace(t0, t1, N)
    use_grid = True
    prob_sol = ode_filter_and_smoother(y0, f, f_args, t0, t1, q, initial_stepsize,
                                       adaptive_stepsize=adaptive_stepsize,
                                       approximation_order=approximation_order,
                                       saveat=grid if use_grid else None,
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
    ys = prob_sol["ys"]
    y_stddevs = prob_sol["stddevs"]

    # TODO good plotting
    plt.plot(sol.ts, sol.ys, label="Tsit5", linewidth=4, color="black")
    plt.plot(ts, ys, label=f"{label_prefix} mean", linewidth=2, color="green")
    plt.fill_between(ts, ys[:, 0] - y_stddevs[:, 0], ys[:, 0] + y_stddevs[:, 0], label=f"{label_prefix} cov", linewidth=1, color="green", alpha=0.3)
    plt.fill_between(ts, ys[:, 1] - y_stddevs[:, 1], ys[:, 1] + y_stddevs[:, 1], label=f"{label_prefix} cov", linewidth=1, color="green", alpha=0.3)
    plt.legend()
    plt.show()
