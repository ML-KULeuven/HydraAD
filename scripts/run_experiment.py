import argparse
import json
import pathlib
from datetime import datetime
from typing import Any, Literal, get_args

import pandas as pd
from dtaianomaly.evaluation import Metric
from dtaianomaly.workflow import Workflow

from hydraad import HydraAD, TSBADLoader, VolumeUnderPR_TSBAD

_TUNE_KWARGS = {
    "window_size": "fft",
    "periodicity": 1.0,
    "overlap_rate": 0.25,
    "quantile": 0.8,
    "scaling": "sparse",
    "n_kernels": 8,
    "n_groups": 64,
    "dilations": "auto",
    "first_order_difference": True,
    "hard_counts": True,
    "soft_counts": True,
    "kernel_size": 9,
    "relative_n_channels": 0.5,
    "max_n_channels": None,
}

_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
_PERIODICITY = [0.5, 1.0, 2.0, 3.0]
_OVERLAP_RATE = [0.0, 0.1, 0.25, 0.5, 1.0]
_SCALING: list[Literal["sparse", "standard", "none"]] = ["sparse", "standard", "none"]

_N_GROUPS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
_N_KERNELS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
_DILATIONS = ["auto", [1], [1, 2, 4]]
_FIRST_ORDER_DIFFERENCE = [True, False]
_COUNTS = [(True, True), (True, False), (False, True)]
_N_CHANNELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
_KERNEL_SIZE = [7, 9, 11]

_DIMENSION_TYPE = Literal["U", "M"]
_SUBSET_TYPE = Literal["Tuning", "Eva"]
_PARAMETRIZATION_TYPE = Literal[
    "default",
    "periodicity",
    "quantiles",
    "overlap_rate",
    "scaling",
    "number_kernels",
    "dilations",
    "first_order_difference",
    "counts",
    "kernel_size",
    "n_channels",
]


def setup_experiment() -> dict[str, Any]:
    # Set up the script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default="../",
        help="Path (potentially relative) to the root of the project, which contains `environment.json`.",
    )
    parser.add_argument(
        "--dimension",
        choices=get_args(_DIMENSION_TYPE),
        required=True,
        help="Specify the dimension: 'U' for univariate or 'M' for multivariate.",
    )
    parser.add_argument(
        "--subset",
        choices=get_args(_SUBSET_TYPE),
        required=True,
        help="Specify the subset of time series to use.",
    )
    parser.add_argument(
        "--parameterization",
        choices=get_args(_PARAMETRIZATION_TYPE),
        default="default",
        help="Which preconfigured parameterization that should be used.",
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

    # Load the additional arguments in the environment
    env["dimension"] = args.dimension
    env["subset"] = args.subset
    env["parameterization"] = args.parameterization
    env["gpu"] = args.gpu

    return env


def initialize_dataloaders(
    path: str, dimension: _DIMENSION_TYPE, subset: _SUBSET_TYPE
) -> list[TSBADLoader]:
    return [
        TSBADLoader(f"{path}/TSB-AD-{dimension}/{file_name}")
        for file_name in pd.read_csv(
            f"{path}/metadata/TSB-AD-{dimension}-{subset}.csv"
        )["file_name"]
    ]


def initialize_detectors(
    parameterization: _PARAMETRIZATION_TYPE,
    dimension: _DIMENSION_TYPE,
    subset: _SUBSET_TYPE,
    gpu: bool,
    debug: bool,
) -> list[HydraAD]:
    if debug:
        print("DEBUG: using 2 seeds for each detector configuration")
        n_seeds = 2
    elif subset == "Tuning":
        print("INFO: using 10 seeds for each detector configuration on tuning set")
        n_seeds = 10
    else:
        print("INFO: using 30 seeds for each detector configuration on evaluation set")
        n_seeds = 30

    if parameterization == "default":
        return [
            HydraAD(
                window_size=_TUNE_KWARGS["window_size"],
                device="cuda" if gpu else "cpu",
                seed=seed,
            )  # type: ignore
            for seed in range(n_seeds)
        ]

    kwargs = _TUNE_KWARGS.copy()

    if parameterization == "quantiles":
        kwargs.pop("quantile")
        return [
            HydraAD(
                quantile=quantile, device="cuda" if gpu else "cpu", seed=seed, **kwargs
            )  # type: ignore
            for quantile in _QUANTILES
            for seed in range(n_seeds)
        ]

    if parameterization == "periodicity":
        kwargs.pop("periodicity")
        return [
            HydraAD(
                periodicity=periodicity,
                device="cuda" if gpu else "cpu",
                seed=seed,
                **kwargs,
            )  # type: ignore
            for periodicity in _PERIODICITY
            for seed in range(n_seeds)
        ]

    if parameterization == "overlap_rate":
        kwargs.pop("overlap_rate")
        return [
            HydraAD(
                overlap_rate=overlap_rate,
                device="cuda" if gpu else "cpu",
                seed=seed,
                **kwargs,
            )  # type: ignore
            for overlap_rate in _OVERLAP_RATE
            for seed in range(n_seeds)
        ]

    if parameterization == "scaling":
        kwargs.pop("scaling")
        return [
            HydraAD(
                scaling=scaling, device="cuda" if gpu else "cpu", seed=seed, **kwargs
            )  # type: ignore
            for scaling in _SCALING
            for seed in range(n_seeds)
        ]

    if parameterization == "number_kernels":
        kwargs.pop("n_groups")
        kwargs.pop("n_kernels")
        return [
            HydraAD(
                n_groups=n_groups,
                n_kernels=n_kernels,
                device="cuda" if gpu else "cpu",
                seed=seed,
                **kwargs,
            )  # type: ignore
            for n_groups in _N_GROUPS
            for n_kernels in _N_KERNELS
            for seed in range(n_seeds)
            if 8 <= n_groups * n_kernels <= 2048
        ]

    if parameterization == "dilations":
        kwargs.pop("dilations")
        return [
            HydraAD(
                dilations=dilations,
                device="cuda" if gpu else "cpu",
                seed=seed,
                **kwargs,
            )  # type: ignore
            for dilations in _DILATIONS
            for seed in range(n_seeds)
        ]

    if parameterization == "first_order_difference":
        kwargs.pop("first_order_difference")
        return [
            HydraAD(
                first_order_difference=first_order_difference,
                device="cuda" if gpu else "cpu",
                seed=seed,
                **kwargs,
            )  # type: ignore
            for first_order_difference in _FIRST_ORDER_DIFFERENCE
            for seed in range(n_seeds)
        ]

    if parameterization == "counts":
        kwargs.pop("hard_counts")
        kwargs.pop("soft_counts")
        return [
            HydraAD(
                hard_counts=hard_counts,
                soft_counts=soft_counts,
                device="cuda" if gpu else "cpu",
                seed=seed,
                **kwargs,  # type: ignore
            )
            for hard_counts, soft_counts in _COUNTS
            for seed in range(n_seeds)
        ]

    if parameterization == "kernel_size":
        kwargs.pop("kernel_size")
        return [
            HydraAD(
                kernel_size=kernel_size,
                device="cuda" if gpu else "cpu",
                seed=seed,
                **kwargs,
            )  # type: ignore
            for kernel_size in _KERNEL_SIZE
            for seed in range(n_seeds)
        ]

    if parameterization == "n_channels":
        assert dimension == "M", (
            "The 'n_channels' parameterization is only valid for multivariate time series!"
        )

        kwargs.pop("relative_n_channels")
        return [
            HydraAD(
                relative_n_channels=relative_n_channels,
                device="cuda" if gpu else "cpu",
                seed=seed,
                **kwargs,
            )  # type: ignore
            for relative_n_channels in _N_CHANNELS
            for seed in range(n_seeds)
        ]

    raise ValueError(f"Undefined parametrization given: '{parameterization}'!")


def initialize_metrics(subset: _SUBSET_TYPE) -> Metric:
    if subset == "Tuning":
        return VolumeUnderPR_TSBAD(threshold=75)
    else:
        return VolumeUnderPR_TSBAD()


def save_results(
    path: str,
    dimension: Literal["U", "M"],
    subset: _SUBSET_TYPE,
    parametrization: _PARAMETRIZATION_TYPE,
    debug: bool,
    df: pd.DataFrame,
) -> None:
    # The path to the raw results directory
    base_path = pathlib.Path(path).resolve() / "raw" / parametrization
    if not base_path.exists():
        print(f"INFO: Creating directory {base_path}")
        base_path.mkdir(parents=True)

    # Path to the results file
    results_path = (
        base_path
        / f"{'DEBUG_' if debug else ''}TSB-AD-{dimension}-{subset}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    )

    print(f"INFO: Saving the results to: {results_path}")

    # Saving the results
    df.to_csv(results_path, index=False)


def main():
    # Setup experiment
    env = setup_experiment()

    # Setup workflow
    dataloaders = initialize_dataloaders(
        env["path_to_data"], env["dimension"], env["subset"]
    )
    detectors = initialize_detectors(
        env["parameterization"],
        env["dimension"],
        env["subset"],
        env["gpu"],
        env["debug"],
    )
    metrics = initialize_metrics(env["subset"])

    # Debugging
    if env["debug"]:
        nb_dataloaders = 4
        dataloaders = dataloaders[:nb_dataloaders]
        print(f"DEBUG: experiment is applied to only {nb_dataloaders} time series")

    # Run workflow
    workflow = Workflow(
        dataloaders=dataloaders,
        metrics=metrics,
        detectors=detectors,
        n_jobs=env["n_jobs"],
        show_progress=True,
    )
    results = workflow.run(default_window_size=64)

    # Check if there were any errors
    if "Error file" in results.columns:
        print(
            f"INFO: Workflow finished, but some errors occurred ({results['Error file'].count()}/{results.shape[0]})!"
        )
    else:
        print("INFO: Workflow finished successfully!")

    # Save results
    save_results(
        env["path_to_results"],
        env["dimension"],
        env["subset"],
        env["parameterization"],
        env["debug"],
        results,
    )


if __name__ == "__main__":
    main()
