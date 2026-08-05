import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def bootstrap_mean(values, rng, n_bootstrap=10000):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    bootstrap_means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        bootstrap_means[i] = sample.mean()

    mean = values.mean()
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])

    return mean, lower, upper


def format_bootstrap(result):
    mean, lower, upper = result
    return f"{mean:.4f} [{lower:.4f}, {upper:.4f}]"


def plot_error_bars(results, configurations, columns, output_path, title):
    y = np.arange(len(configurations))
    offsets = np.linspace(-0.3, 0.3, len(columns))

    plt.figure(figsize=(12, 7))

    for column_index, column in enumerate(columns):
        means = np.array([
            result[column_index][0]
            for result in results
        ])
        lowers = np.array([
            result[column_index][1]
            for result in results
        ])
        uppers = np.array([
            result[column_index][2]
            for result in results
        ])

        errors = np.vstack([
            means - lowers,
            uppers - means,
        ])

        plt.errorbar(
            means,
            y + offsets[column_index],
            xerr=errors,
            fmt="o",
            capsize=3,
            label=column,
        )

    # Faint horizontal separators between configurations.
    for i in range(len(configurations) - 1):
        plt.axhline(
            i + 0.5,
            linewidth=0.8,
            alpha=0.2,
        )

    plt.yticks(y, configurations)
    plt.xlabel("Rate")
    plt.title(title)
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-folder",
        type=str,
        default="./evaluation",
        help="Folder where evaluation plots are saved.",
    )
    args = parser.parse_args()

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    main_dir = Path(__file__).resolve().parent.parent
    dataset_folder = os.path.join(main_dir, "dataset")

    directory = Path(dataset_folder)

    directories = [path for path in directory.iterdir() if path.is_dir()]
    configurations = [d.name for d in directories]

    results = []
    results_errs = []

    numeric_results = []
    numeric_results_errs = []

    rng = np.random.default_rng(42)

    for d in directories:
        meta_file = os.path.join(d, "metadata.jsonl")
        err_file = os.path.join(d, "error.jsonl")

        print(meta_file)

        accept_values = [[] for _ in range(6)]

        error_count = {
            "code_execution": 0,
            "code_regeneration": 0,
        }

        all_count = 0
        all_iters_count = 0

        with open(meta_file, "r", encoding="utf-8") as f:
            for line in f:
                graph = json.loads(line)

                for i in range(6):
                    accepted = (
                        len(graph["images"]) < i + 1
                        or graph["images"][i]["accept"]
                    )
                    accept_values[i].append(int(accepted))

                all_count += 1
                all_iters_count += max(len(graph["images"]) - 1, 0)

        with open(err_file, "r", encoding="utf-8") as f:
            for line in f:
                err = json.loads(line)

                for k in error_count:
                    if err["stage"] == k:
                        error_count[k] += 1

        execution_errors = error_count["code_execution"]
        regeneration_errors = error_count["code_regeneration"]

        if execution_errors > all_count:
            raise ValueError(
                f"{d.name}: more code_execution errors "
                f"({execution_errors}) than graphs ({all_count})."
            )

        if regeneration_errors > all_iters_count:
            raise ValueError(
                f"{d.name}: more code_regeneration errors "
                f"({regeneration_errors}) than iterations "
                f"({all_iters_count})."
            )

        execution_values = (
            [1] * execution_errors
            + [0] * (all_count - execution_errors)
        )

        regeneration_values = (
            [1] * regeneration_errors
            + [0] * (all_iters_count - regeneration_errors)
        )

        acceptance_bootstrap = [
            bootstrap_mean(values, rng)
            for values in accept_values
        ]

        error_bootstrap = [
            bootstrap_mean(execution_values, rng),
            bootstrap_mean(regeneration_values, rng),
        ]

        numeric_results.append(acceptance_bootstrap)
        numeric_results_errs.append(error_bootstrap)

        results.append([
            format_bootstrap(result)
            for result in acceptance_bootstrap
        ])

        results_errs.append([
            format_bootstrap(result)
            for result in error_bootstrap
        ])

    acceptance_columns = [
        f"accept@{i}"
        for i in range(6)
    ]

    error_columns = [
        "code_execution",
        "code_regeneration",
    ]

    df = pd.DataFrame(
        results,
        index=configurations,
        columns=acceptance_columns,
    )

    df_err = pd.DataFrame(
        results_errs,
        index=configurations,
        columns=error_columns,
    )

    print("Acceptance rates with 95% bootstrap confidence intervals:\n")
    print(df.to_markdown())

    print("\nError rates with 95% bootstrap confidence intervals:\n")
    print(df_err.to_markdown())

    plot_error_bars(
        numeric_results,
        configurations,
        acceptance_columns,
        os.path.join(output_folder, "acceptance_rates.png"),
        "Acceptance rates with 95% bootstrap confidence intervals",
    )

    plot_error_bars(
        numeric_results_errs,
        configurations,
        error_columns,
        os.path.join(output_folder, "error_rates.png"),
        "Error rates with 95% bootstrap confidence intervals",
    )