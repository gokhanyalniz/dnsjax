#!/usr/bin/env python3
import os
import sys
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
    from sharding import get_zeros, sharding
    from stats import get_stats
    from timestep import iterate_correction, predict_and_correct, stepper
    from velocity import correct_velocity

    if params.init.start_from_laminar:
        velocity_spec = get_zeros(
            shape=(3, *sharding.spec_shape),
            dtype=sharding.complex_type,
            in_sharding=sharding.spec_vector_shard,
            out_sharding=sharding.spec_vector_shard,
        )
        if force.on:
            velocity_spec = velocity_spec.at[force.forced_modes].add(
                force.unit_force * force.laminar_amplitude
            )

    elif params.init.snapshot is not None:
        snapshot = jnp.load(params.init.snapshot)["velocity_phys"].astype(
            sharding.phys_type
        )
        velocity_phys = jax.device_put(
            snapshot,
            sharding.phys_vector_shard,
        )
        velocity_spec = phys_to_spec(velocity_phys)

    else:
        sharding.print("Provide an initial condition.")
        sharding.exit(code=1)

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
    c_tot = 0
    dt_first = params.step.dt
    wall_time_now = perf_counter_ns()
    bench_delta = 0
    corrector_compiled = False
    last_error = 0
    last_c = 0
    norm_corrections = {}

    # Call once now not to affect benchmarks later
    stats = get_stats(
        velocity_spec,
        fourier.lapl,
        fourier.metric,
    )

    sharding.print(
        f"t = {t:.2f}",
        *[f"{x}={y:.3e}" for x, y in stats.items()],
    )

    sharding.print("Started timestepping at", datetime.now())

    while (
        t < t_stop
        and wall_time_now - wall_time_start < wall_time_stop
        and last_error < params.step.corrector_tolerance
    ):
        if it == params.init.it0 + 1:
            # Ignore the first hit, probably subject to JIT compilation
            bench_start = perf_counter_ns()

            sharding.print("First iteration over at", datetime.now())

        if (
            params.outs.it_stats is not None
            and it % params.outs.it_stats == 0
            and it > params.init.it0
        ):
            stats = get_stats(
                velocity_spec,
                fourier.lapl,
                fourier.metric,
            )
            c_per_it = c_tot / (it - params.init.it0)

            sharding.print(
                f"t = {t:.2f}",
                *[f"{x}={y:.3e}" for x, y in stats.items()],
                f"c/it = {c_per_it:.2f}",
                f"err = {last_error:.3e}",
                *[f"{x}={y:.3e}" for x, y in norm_corrections.items()]
                if norm_corrections is not None
                else "",
            )

        velocity_spec, rhs_no_lapl, error = predict_and_correct(
            velocity_spec,
            fourier.kvec,
            fourier.inv_lapl,
            fourier.metric,
            stepper.ldt_1,
            stepper.ildt_2,
        )
        c = 0

        while (
            error > params.step.corrector_tolerance
            and c < params.step.max_corrector_iterations
        ):
            if not corrector_compiled:
                # Do not include this in the benchmark
                bench_delta_start = perf_counter_ns()

            velocity_spec, rhs_no_lapl, error = iterate_correction(
                velocity_spec,
                rhs_no_lapl,
                fourier.kvec,
                fourier.inv_lapl,
                fourier.metric,
                stepper.ildt_2,
            )
            c += 1

            if not corrector_compiled:
                bench_delta_stop = perf_counter_ns()
                bench_delta += bench_delta_stop - bench_delta_start
                rhs_tot -= 1
                corrector_compiled = True

        velocity_spec, norm_corrections = correct_velocity(
            velocity_spec, fourier.kvec, fourier.inv_lapl, fourier.metric
        )

        t += params.step.dt
        it += 1
        last_error = error
        last_c = c
        c_tot += c

        if it > params.init.it0:
            # Ignore the first hit, probably subject to JIT compilation
            rhs_tot += c + 2  # 1 rhs per corrector + 2 rhs per predict-correct

        wall_time_now = perf_counter_ns()

    if last_error > params.step.corrector_tolerance:
        sharding.print(
            f"Corrector failed to converge at t={t}, it={it}, c={last_c}, "
            f"with error = {last_error:.3e}."
        )

    sharding.print("Stopped timestepping at", datetime.now())

    wall_time_now = perf_counter_ns()
    alive_time = bench.ns_to_s * (wall_time_now - wall_time_start)
    sharding.print(f"Job has been alive for {alive_time:.2f}s.")
    if it > params.init.it0 + 1:
        wall_time = bench.ns_to_s * (wall_time_now - bench_delta - bench_start)
        wall_time_per_sim_time = wall_time / (t - dt_first - params.init.t0)
        wall_time_per_rhs = wall_time / rhs_tot

        # Useful to know final stats
        stats = get_stats(
            velocity_spec,
            fourier.lapl,
            fourier.metric,
        )
        c_per_it = c_tot / (it - params.init.it0)

        sharding.print(
            f"t = {t:.2f}",
            *[f"{x}={y:.3e}" for x, y in stats.items()],
            f"c/it = {c_per_it:.2f}",
            f"err = {last_error:.3e}",
            *[f"{x}={y:.3e}" for x, y in norm_corrections.items()]
            if norm_corrections is not None
            else "",
        )

        if params.debug.time_functions and main_device:
            pp(bench.timers, sort_dicts=True)

        if sharding.n_devices > 1:
            sharding.print(
                f"Ran for {wall_time:.2f}s with {sharding.n_devices} devices,",
                f"{sharding.n_devices * wall_time:.3e} NP x s:",
                f"{wall_time_per_sim_time:.3e} s/t,",
                f"{sharding.n_devices * wall_time_per_sim_time:.3e} NP x s/t,",
                f"{wall_time_per_rhs:.3e} s/rhs,",
                f"{sharding.n_devices * wall_time_per_rhs:.3e} NP x s/rhs.",
            )
        else:
            sharding.print(
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
    main_device = bool(jax.process_index() == 0)

    if main_device:
        print("Distribution initialized at", datetime.now(), flush=True)
        if params_from_disk:
            print(
                "Loaded parameters.toml, "
                "which override the default parameters. "
                "Command-line arguments will further override "
                "the loaded parameters.",
                flush=True,
            )
        else:
            print(
                "Loaded the default parameters, "
                "as parameters.toml was not found. "
                "Command-line arguments will further override "
                "the default parameters.",
                flush=True,
            )
        print("Final working parameters:")
        if main_device:
            pp(params.model_dump())

        print(
            "Running with the effective resolution:",
            padded_res.nx_padded,
            padded_res.ny_padded,
            padded_res.nz_padded,
            flush=True,
        )

    main()

    print("Shutdown at", datetime.now(), flush=True)
    sys.exit(0)
