import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt
from probabilistic_ode_solver import ode_filter_and_smoother
from diffeqs import linear_vf, lotka_volterra_vf


def main():
    y0 = jnp.array([10.0, 10.0])
    f = lotka_volterra_vf
    f_args = (0.1, 0.02, 0.4, 0.02)
    q = 3
    t0 = 0.0
    t1 = 140.0
    N = 1000
    initial_stepsize = 1e-1
    approximation_order = 0
    apply_smoother = True
    adaptive_stepsize = True
    grid = jnp.linspace(t0, t1, N)
    use_grid = True
    reltol = 1e-3
    prob_sol = ode_filter_and_smoother(y0, f, f_args, t0, t1, q, initial_stepsize,
                                       reltol=reltol,
                                       adaptive_stepsize=adaptive_stepsize,
                                       approximation_order=approximation_order,
                                       saveat=grid if use_grid else None,
                                       apply_smoother=apply_smoother)

    term = diffrax.ODETerm(f)
    solver = diffrax.Tsit5()
    dt0 = 1e-1
    saveat = diffrax.SaveAt(ts=grid)
    sol = diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0,
                              args=f_args,
                              saveat=saveat)
    true_d = f(sol.ts, sol.ys, f_args)

    label_prefix = "Smoothing" if apply_smoother else "Filtering"
    ts = prob_sol["ts"]
    ys = prob_sol["ys"]
    y_stddevs = prob_sol["stddevs"]
    print(f"Steps: rejected: {prob_sol['num_rejected']}, accepted: {prob_sol['num_accepted']}")

    # TODO good plotting
    plt.plot(sol.ts, sol.ys, label="Tsit5", linewidth=2, color="grey")
    plt.plot(ts, ys, label=f"{label_prefix} mean", linewidth=2, color="green")
    plt.fill_between(ts, ys[:, 0] - y_stddevs[:, 0], ys[:, 0] + y_stddevs[:, 0], label=f"{label_prefix} cov", linewidth=1, color="green", alpha=0.3)
    plt.fill_between(ts, ys[:, 1] - y_stddevs[:, 1], ys[:, 1] + y_stddevs[:, 1], label=f"{label_prefix} cov", linewidth=1, color="green", alpha=0.3)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
