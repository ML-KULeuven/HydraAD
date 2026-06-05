import argparse
import json
import pathlib
import time
from datetime import datetime
from typing import Any, Literal, get_args

import numpy as np
import pandas as pd
import torch
import tqdm

from hydraad import HydraAD

_DEFAULT_LENGTH = 250_000
_LENGTHS = [
    1_000,
    5_000,
    10_000,
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
    2_500_000,
    5_000_000,
]
_DEFAULT_DIMENSION = 1
_DIMENSIONS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
_DEFAULT_N_GROUPS = 32
_N_GROUPS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
_DEFAULT_N_KERNELS = 4
_N_KERNELS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
_N_RUNS = 10
_PARAMETER_TO_VARY = Literal[
    "dimension",
    "length",
    "number_kernels",
]


def setup_experiment() -> dict[str, Any]:
    """
    Set up the script arguments.

    Returns
    -------
    env : dict[str, Any]
        Environment configuration dictionary with project settings.
    """
    parser = argparse.ArgumentParser(
        description="Run scalability analysis for HydraAD."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="../",
        help="Path (potentially relative) to the root of the project, which contains `environment.json`.",
    )
    parser.add_argument(
        "--parameter",
        choices=get_args(_PARAMETER_TO_VARY),
        required=True,
        help="Which parameter to vary in the scalability analysis.",
    )
    parser.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use GPU for HydraAD.",
    )
    args = parser.parse_args()

    # Set up the path to the environment variables
    env_path = pathlib.Path(args.path).resolve() / "environment.json"
    print(f"INFO: Loading environment from: {env_path}")
    assert env_path.exists(), (
        f"There is no `{env_path.name}` at '{env_path.parents[0]}'!"
    )

    # Load the environment
    with open(env_path, "r") as file:
        env = json.load(file)

    env["parameter"] = args.parameter
    env["gpu"] = args.gpu

    return env


def generate_random_timeseries(length: int, n_channels: int, seed: int) -> np.ndarray:
    """
    Generate a random time series.

    Generate a random time series of given size, in which each observation is sampled
    uniformly at random from the half-open interval [0, 1).

    Parameters
    ----------
    length : int
        Length of the time series.
    n_channels : int
        Number of channels (dimensions) of the time series.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray
        Random time series of shape (length, n_channels) for multivariate
        or (length,) for univariate.
    """
    rng = np.random.default_rng(seed)
    return rng.random((length, n_channels)).squeeze()


def run_hydraad(
    length: int, n_channels: int, n_groups: int, n_kernels: int, gpu: bool, seed: int
) -> (float, float):
    """
    Run HydraAD on a time series and measure runtime and peak memory usage.

    Parameters
    ----------
    length : int
        Length of the time series.
    n_channels : int
        Number of channels (dimensions) of the time series.
    n_groups : int
        Number of groups for HydraAD.
    n_kernels : int
        Number of kernels for HydraAD.
    gpu : bool
        Whether to use GPU for HydraAD.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    runtime_ms : float
        Runtime in milliseconds.
    peak_memory_mb : float
        Peak memory usage in megabytes.
    """

    X = generate_random_timeseries(length=length, n_channels=n_channels, seed=seed)

    # Create detector with default parameters
    detector = HydraAD(
        window_size=100,
        n_groups=n_groups,
        n_kernels=n_kernels,
        device="cuda" if gpu else "cpu",
        seed=seed,
    )

    # Start memory tracking
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time_ns()

    # Run fit
    detector.fit(X)

    # Get predictions
    detector.decision_function(X)

    # Get timing and memory
    runtime = time.time_ns() - start_time
    runtime_ms = runtime / 1e6  # Convert to milliseconds
    peak_bytes = torch.cuda.max_memory_allocated()
    peak_memory_mb = peak_bytes / (1024 * 1024)

    return runtime_ms, peak_memory_mb


def save_results(
    path: str,
    debug: bool,
    parameter: _PARAMETER_TO_VARY,
    gpu: bool,
    df: pd.DataFrame,
) -> None:
    """
    Save scalability analysis results to a CSV file.

    Parameters
    ----------
    path : str
        Path to the results directory.
    debug : bool
        Whether debug mode is enabled.
    parameter : _PARAMETER_TO_VARY
        The parameter that was varied in the scalability analysis.
    gpu : bool
        Whether GPU was used for HydraAD.
    df : pd.DataFrame
        Results dataframe to save.
    """
    base_path = pathlib.Path(path).resolve() / "raw" / "scalability" / parameter
    if not base_path.exists():
        print(f"INFO: Creating directory {base_path}")
        base_path.mkdir(parents=True)

    # Path to the results file
    results_path = (
        base_path
        / f"{'DEBUG_' if debug else ''}{'gpu_' if gpu else 'cpu_'}{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    )
    print(f"INFO: Saving the results to: {results_path}")

    # Saving the results
    df.to_csv(results_path, index=False)


def main() -> None:
    """Run the scalability analysis."""
    # Setup experiment
    env = setup_experiment()

    # Determine dataset sizes based on debug mode
    if env["debug"]:
        lengths = [1000, 5000]
        dimensions = [1, 4]
        n_groups = [4]
        n_kernels = [32]
        n_runs = 2
        print("DEBUG: using reduced dataset")
    else:
        lengths = _LENGTHS
        dimensions = _DIMENSIONS
        n_groups = _N_GROUPS
        n_kernels = _N_KERNELS
        n_runs = _N_RUNS

    # Set up the jobs to run
    jobs = []
    if "length" == env["parameter"]:
        jobs = [
            (length, _DEFAULT_DIMENSION, _DEFAULT_N_GROUPS, _DEFAULT_N_KERNELS)
            for length in lengths
        ]
    elif "dimension" == env["parameter"]:
        jobs = [
            (_DEFAULT_LENGTH, dimension, _DEFAULT_N_GROUPS, _DEFAULT_N_KERNELS)
            for dimension in dimensions
        ]
    elif "number_kernels" == env["parameter"]:
        jobs = [
            (_DEFAULT_LENGTH, _DEFAULT_DIMENSION, g, k)
            for g in n_groups
            for k in n_kernels
            if 8 <= g * k <= 2048
        ]
    jobs = [(*job, run) for job in jobs for run in range(n_runs)]

    # Warm runs to compile everything
    for i in range(3):
        run_hydraad(5_000, 3, 32, 4, env["gpu"], seed=i)

    # Run the experiments and collect results
    results = []
    for length, n_channels, g, k, run in tqdm.tqdm(jobs):
        # Run HydraAD
        runtime, peak_memory = run_hydraad(
            length, n_channels, g, k, env["gpu"], seed=run
        )

        # Compute statistics
        results.append(
            {
                "Length": length,
                "Dimension": n_channels,
                "n_groups": g,
                "n_kernels": k,
                "Seed": run,
                "Run time [ms]": runtime,
                "Peak memory [Mb]": peak_memory,
            }
        )

    # Save results
    save_results(
        env["path_to_results"],
        env["debug"],
        env["parameter"],
        env["gpu"],
        pd.DataFrame(results),
    )

    print("INFO: Scalability analysis completed successfully!")


if __name__ == "__main__":
    main()
