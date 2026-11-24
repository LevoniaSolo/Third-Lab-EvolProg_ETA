import csv
import json
import platform
from fractions import Fraction

import matplotlib
import numpy as np

from data import N_FEATURES, REDUNDANT_WEIGHTS, ValidationFitness, error_count, generate_dataset
from nsga2 import pareto_points, search
from plots import plot_history, plot_pareto


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize_runs(run_summaries):
    rows = []
    ddof = 0
    for metric in ("hypervolume", "best_validation_accuracy", "min_features"):
        values = np.array([run[metric] for run in run_summaries], dtype=float)
        rows.append({
            "metric": metric,
            "runs": values.size,
            "min": values.min(),
            "mean": values.mean(),
            "median": np.median(values),
            "std": values.std(ddof=ddof),
            "max": values.max(),
            "ddof": ddof,
        })
    return rows


def save_dataset(output, data, config):
    feature_names = [f"x{i + 1:02d}" for i in range(N_FEATURES)]
    split_names = np.empty(data.labels.size, dtype=object)
    split_counts = {}
    for name in ("train", "validation", "test"):
        indices = getattr(data, name)
        split_names[indices] = name
        split_counts[name] = {"size": len(indices), "class_1": int(data.labels[indices].sum())}
    with (output / "dataset.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["row_id", "split", "label", *feature_names])
        for i, features in enumerate(data.features):
            writer.writerow([i, split_names[i], data.labels[i], *features])
    save_json(output / "dataset_metadata.json", {
        "generation": config,
        "redundant_weights": REDUNDANT_WEIGHTS,
        "columns": dict(zip(feature_names, data.feature_types)),
        "split_counts": split_counts,
        "split_row_ids": {
            name: getattr(data, name).tolist() for name in split_counts
        },
    })


def score_row(mask, objectives, validation_size):
    errors, count = objectives
    return {
        "mask": "".join(map(str, mask)),
        "n_features": count,
        "validation_errors": errors,
        "validation_error": errors / validation_size,
        "validation_accuracy": 1 - errors / validation_size,
    }


def representative_masks(scores, front):
    # Use the lexicographically smallest mask when several share the same point.
    by_point = {}
    points = set(front)
    for mask, objectives in scores.items():
        if objectives in points and (objectives not in by_point or mask < by_point[objectives]):
            by_point[objectives] = mask
    return [by_point[point] for point in front]


def select_masks(scores, front):
    masks = representative_masks(scores, front)
    selected = {"min_features": masks[0]}
    if len(front) >= 3:
        low_error, high_count = front[-1]
        high_error, low_count = front[0]

        def distance_to_ideal(index):
            errors, count = front[index]
            return ((errors - low_error) / (high_error - low_error)) ** 2 + (
                (count - low_count) / (high_count - low_count)) ** 2

        index = min(range(1, len(front) - 1), key=distance_to_ideal)
        selected["compromise"] = masks[index]
    if len(front) >= 2:
        selected["max_quality"] = masks[-1]
    return selected


def weight_interval(point, front, validation_size):
    """Find the closed interval of error weights for which point is optimal."""
    lower, upper = Fraction(0), Fraction(1)
    for other in front:
        intercept = Fraction(point[1] - other[1], N_FEATURES - 1)
        slope = Fraction(point[0] - other[0], validation_size) - intercept
        if slope > 0:
            upper = min(upper, -intercept / slope)
        elif slope < 0:
            lower = max(lower, -intercept / slope)
        elif intercept > 0:
            return None
    return (lower, upper) if lower <= upper else None


def weighted_rows(scores, front, validation_size):
    rows = []
    for point, mask in zip(front, representative_masks(scores, front)):
        interval = weight_interval(point, front, validation_size)
        rows.append({
            **score_row(mask, point, validation_size),
            "supported": interval is not None,
            "unique_optimum_for_some_weight": interval is not None and interval[0] < interval[1],
            "min_error_weight": str(interval[0]) if interval else "",
            "max_error_weight": str(interval[1]) if interval else "",
        })
    return rows


def evaluate_final(data, selected, scores, fitness):
    full_mask = (1,) * N_FEATURES
    fit_rows = np.concatenate((data.train, data.validation))
    rows = []
    for role, mask in [*selected.items(), ("all_features", full_mask)]:
        objectives = fitness(full_mask) if role == "all_features" else scores[mask]
        errors = error_count(data.features[fit_rows], data.labels[fit_rows],
                             data.features[data.test], data.labels[data.test],
                             mask, fitness.ridge)
        rows.append({
            "role": role,
            **score_row(mask, objectives, len(data.validation)),
            "selected_features": " ".join(f"x{i + 1:02d}" for i, bit in enumerate(mask) if bit),
            "test_errors": errors,
            "test_error": errors / len(data.test),
            "test_accuracy": 1 - errors / len(data.test),
        })
    return rows


def run_experiment(config, output):
    search_config = config["search"]
    runs = search_config["runs"]
    if runs < 1:
        raise ValueError("At least one independent run is required.")
    data = generate_dataset(config["data"])
    validation_size = len(data.validation)
    fitness = ValidationFitness(data.features[data.train], data.labels[data.train],
                                data.features[data.validation], data.labels[data.validation],
                                config["classifier"]["ridge"])
    seeds = np.random.SeedSequence(search_config["seed"]).spawn(runs)
    seeds = [int(seed.generate_state(1, dtype=np.uint64)[0]) for seed in seeds]
    histories = []
    run_summaries = []
    mask_rows = []
    front_rows = []
    weights = []
    pooled_scores = {}
    for run, seed in enumerate(seeds, start=1):
        result = search(fitness, N_FEATURES, validation_size, seed, search_config)
        pooled_scores.update(result.scores)
        histories.extend({"run": run, "seed": seed, **row} for row in result.history)
        archive_front = pareto_points(result.scores.values())
        population_front = pareto_points(result.scores[mask] for mask in result.population)
        run_summaries.append({
            "run": run,
            "seed": seed,
            **result.history[-1],
            "final_population_pareto_points": len(population_front),
        })
        mask_rows.extend({"run": run, **score_row(mask, point, validation_size)}
                         for mask, point in result.scores.items())
        for scope, points, masks in (
            ("archive", archive_front, result.scores),
            ("final_population", population_front,
             {mask: result.scores[mask] for mask in result.population}),
        ):
            front_rows.extend({"run": run, "scope": scope, **score_row(mask, masks[mask], validation_size)}
                              for mask in representative_masks(masks, points))
        weights.extend({"scope": str(run), **row}
                       for row in weighted_rows(result.scores, archive_front, validation_size))
        print(f"Run {run}/{runs}: seed={seed}, unique masks={len(result.scores)}, "
              f"Pareto points={len(archive_front)}", flush=True)

    front = pareto_points(pooled_scores.values())
    front_set = set(front)
    pooled_rows = [score_row(mask, point, validation_size)
                   for mask, point in sorted(pooled_scores.items(), key=lambda item: (item[1][1], item[0]))
                   if point in front_set]
    pooled_weights = weighted_rows(pooled_scores, front, validation_size)
    weights.extend({"scope": "pooled", **row} for row in pooled_weights)
    selected = select_masks(pooled_scores, front)
    final_rows = evaluate_final(data, selected, pooled_scores, fitness)
    if len(selected) < 3:
        print(f"Only {len(front)} distinct Pareto points found. "
              "The full set of three representative roles is unavailable.", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    save_dataset(output, data, config["data"])
    save_json(output / "experiment.json", {
        "config": config,
        "run_seeds": seeds,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__},
        "pooled_unique_masks": len(pooled_scores),
        "pooled_pareto_points": len(front),
        "pooled_pareto_masks": len(pooled_rows),
        "weighted_sum_supported_points": sum(row["supported"] for row in pooled_weights),
        "missing_representative_roles": [
            role for role in ("min_features", "compromise", "max_quality") if role not in selected
        ],
    })
    save_csv(output / "runs.csv", run_summaries)
    save_csv(output / "summary.csv", summarize_runs(run_summaries))
    save_csv(output / "history.csv", histories)
    save_csv(output / "evaluated_masks.csv", mask_rows)
    save_csv(output / "run_fronts.csv", front_rows)
    save_csv(output / "pareto_front.csv", pooled_rows)
    save_csv(output / "weighted_sum.csv", weights)
    save_csv(output / "selected_masks.csv", final_rows)
    plot_pareto(pooled_weights, final_rows, output / "pareto.png")
    plot_history(histories, output / "history.png")
