#!/usr/bin/env python3
import os
from datetime import datetime
from pathlib import Path
from pprint import pp
from time import perf_counter_ns

from pydantic_settings import CliApp

from parameters import (
    CLIParameters,
    padded_res,
    params,
    read_parameters,
    update_parameters,
)


def main():

    from jax import numpy as jnp

    import bench
    from operators import fourier, phys_to_spec
    from rhs import force
    from sharding import sharding
    from stats import get_stats
    from timestep import predict_and_correct, stepper
    from velocity import correct_velocity, get_zero_velocity_spec

    if params.init.start_from_laminar:
        velocity_spec = get_zero_velocity_spec(ndims=3)
        if force.on:
            velocity_spec = velocity_spec.at[force.ic_f].add(
                force.laminar_state
            )

    elif params.init.snapshot is not None:
        velocity_phys = jax.device_put(
            jnp.load(params.init.snapshot)["velocity_phys"].astype(
                sharding.phys_type
            ),
            sharding.phys_shard,
        )
        velocity_spec = phys_to_spec(velocity_phys, fourier.active_modes)

    else:
        main_print("Provide an initial condition.")
        return

    wall_time_stop = (
        jnp.inf
        if params.stop.max_wall_time is None
        else int(params.stop.max_wall_time.total_seconds() / bench.ns_to_s)
    )

    t_stop = (
        jnp.inf
        if params.stop.max_sim_time is None
        else params.stop.max_sim_time
    )

    it = params.init.it0
    t = params.init.t0

    rhs_tot = 0
    dt_first = params.step.dt
    wall_time_now = perf_counter_ns()
    last_error = 0
    last_c = 0

    # Call once now not to affect benchmarks later
    stats = get_stats(
        velocity_spec,
        force.laminar_state,
        fourier.lapl,
    )

    main_print(
        f"t = {t:.2f}",
        *[f"{x}={y:.6e}" for x, y in stats.items()],
    )

    main_print("Started timestepping at", datetime.now())

    while (
        t < t_stop
        and wall_time_now - wall_time_start < wall_time_stop
        and last_error < params.step.corrector_tolerance
    ):
        if it == params.init.it0 + 1:
            # Ignore the first hit, probably subject to JIT compilation
            bench_start = perf_counter_ns()

            main_print("First iteration over at", datetime.now())

        if (
            params.outs.it_stats is not None
            and it % params.outs.it_stats == 0
            and it > params.init.it0
        ):
            stats = get_stats(
                velocity_spec,
                force.laminar_state,
                fourier.lapl,
            )
            main_print(
                f"t = {t:.2f}",
                *[f"{x}={y:.6e}" for x, y in stats.items()],
                f"c/it = {rhs_tot / (it - params.init.it0):.2f}",
                f"err = {last_error:.3e}",
            )

        # error = jnp.inf
        c = 1

        velocity_spec, error = predict_and_correct(
            velocity_spec,
            force.laminar_state,
            fourier.nabla,
            fourier.inv_lapl,
            fourier.active_modes,
            stepper.ldt_1,
            stepper.ildt_2,
        )
        c = 1

        while (
            error > params.step.corrector_tolerance
            and c < params.step.max_corrector_iterations
        ):
            velocity_spec, error = predict_and_correct(
                velocity_spec,
                force.laminar_state,
                fourier.nabla,
                fourier.inv_lapl,
                fourier.active_modes,
                stepper.ldt_1,
                stepper.ildt_2,
            )
            print(t, c, error)
            c += 1

        velocity_spec = correct_velocity(
            velocity_spec, fourier.nabla, fourier.inv_lapl, fourier.zero_mean
        )

        t += params.step.dt
        it += 1
        last_error = error
        last_c = c

        if it > params.init.it0:
            # Ignore the first hit, probably subject to JIT compilation
            rhs_tot += c * 2  # 2 rhs calculations per prediction-correction

        wall_time_now = perf_counter_ns()

    if last_error > params.step.corrector_tolerance:
        main_print(
            f"Corrector failed to converge at t={t}, it={it}, c={last_c}, "
            f"with error = {last_error:.3e}."
        )

    main_print("Stopped timestepping at", datetime.now())

    wall_time_now = perf_counter_ns()
    alive_time = bench.ns_to_s * (wall_time_now - wall_time_start)
    main_print(f"Job has been alive for {alive_time:.2f}s.")
    if it > params.init.it0 + 1:
        wall_time = bench.ns_to_s * (wall_time_now - bench_start)
        wall_time_per_sim_time = wall_time / (t - dt_first - params.init.t0)
        wall_time_per_rhs = wall_time / rhs_tot

        # Useful to know final stats
        stats = get_stats(
            velocity_spec,
            force.laminar_state,
            fourier.lapl,
        )
        main_print(
            f"t = {t:.2f}",
            *[f"{x}={y:.6e}" for x, y in stats.items()],
            f"c/it = {rhs_tot / (it - 1 - params.init.it0):.2f}",
            f"err = {last_error:.3e}",
        )

        if params.debug.time_functions and main_device:
            pp(bench.timers, sort_dicts=True)

        if sharding.n_devices > 1:
            main_print(
                f"Ran for {wall_time:.2f}s with {sharding.n_devices} devices,",
                f"{sharding.n_devices * wall_time:.3e} NP x s:",
                f"{wall_time_per_sim_time:.3e} s/t,",
                f"{sharding.n_devices * wall_time_per_sim_time:.3e} NP x s/t,",
                f"{wall_time_per_rhs:.3e} s/rhs,",
                f"{sharding.n_devices * wall_time_per_rhs:.3e} NP x s/rhs.",
            )
        else:
            main_print(
                f"Ran for {wall_time:.2f}s with 1 device.",
                f"{wall_time_per_sim_time:.3e} s/t,",
                f"{wall_time_per_rhs:.3e} s/rhs.",
            )


if __name__ == "__main__":
    print("Alive at", datetime.now(), flush=True)
    wall_time_start = perf_counter_ns()
    params_cli = CliApp.run(CLIParameters)

    params_file = Path("parameters.toml")
    params_from_disk = False
    if Path.is_file(params_file):
        params_from_disk = True
        params_in = read_parameters(params_file)
        update_parameters(params_in)

    update_parameters(params_cli)
    padded_res.set_padded_resolution(params)

    if params.dist.platform == "cpu":
        os.environ["XLA_FLAGS"] = (
            "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
        )
        os.environ["NPROC"] = "1"

    import jax

    jax.config.update("jax_enable_x64", params.res.double_precision)
    jax.config.update("jax_platforms", params.dist.platform)
    jax.distributed.initialize()

    rank = jax.process_index()
    main_device = False
    if rank == 0:
        main_device = True

    def main_print(*args, **kwargs):
        if main_device:
            print(*args, **kwargs, flush=True)

    main_print("JAX initialized at", datetime.now())
    if params_from_disk:
        main_print(
            "Loaded parameters.toml, "
            "which override the default parameters. "
            "Command-line arguments will further override "
            "the loaded parameters.",
        )
    else:
        main_print(
            "Loaded the default parameters, "
            "as parameters.toml was not found. "
            "Command-line arguments will further override "
            "the default parameters.",
        )
    main_print("Final working parameters:")
    if main_device:
        pp(params.model_dump())

    main()

    jax.distributed.shutdown()
    main_print("Shutdown at", datetime.now())
