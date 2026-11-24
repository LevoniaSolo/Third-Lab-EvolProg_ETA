from dataclasses import dataclass

import numpy as np


Mask = tuple[int, ...]
Objectives = tuple[int, int]


@dataclass
class SearchResult:
    population: list[Mask]
    scores: dict[Mask, Objectives]
    history: list[dict]


def dominates(first, second):
    pairs = tuple(zip(first, second, strict=True))
    return all(a <= b for a, b in pairs) and any(a < b for a, b in pairs)


def nondominated_sort(objectives):
    """Return fronts of population indices, starting with rank zero."""
    if not len(objectives):
        return []
    values = np.asarray(objectives)
    no_worse = values[:, None, :] <= values[None, :, :]
    better = values[:, None, :] < values[None, :, :]
    dominance = no_worse.all(axis=2) & better.any(axis=2)
    incoming = dominance.sum(axis=0)
    front = np.flatnonzero(incoming == 0)
    fronts = []
    while front.size:
        fronts.append(front.tolist())
        incoming[front] = -1
        incoming -= dominance[front].sum(axis=0)
        front = np.flatnonzero(incoming == 0)
    return fronts


def crowding_distance(objectives, front):
    distances = dict.fromkeys(front, 0.0)
    if len(front) <= 2:
        return dict.fromkeys(front, float("inf"))
    for objective in range(len(objectives[front[0]])):
        ordered = sorted(front, key=lambda i: objectives[i][objective])
        values = [objectives[i][objective] for i in ordered]
        span = values[-1] - values[0]
        if span == 0:
            continue
        distances[ordered[0]] = distances[ordered[-1]] = float("inf")
        for i in range(1, len(ordered) - 1):
            distances[ordered[i]] += (values[i + 1] - values[i - 1]) / span
    return distances


def repair(mask, rng):
    if any(mask):
        return mask
    index = int(rng.integers(len(mask)))
    return tuple(int(i == index) for i in range(len(mask)))


def initialize(size, n_features, rng):
    population = []
    for _ in range(size):
        mask = tuple(rng.integers(2, size=n_features).tolist())
        population.append(repair(mask, rng))
    return population


def crossover(first, second, probability, rng):
    if rng.random() >= probability or len(first) < 2:
        return first, second
    cut = int(rng.integers(1, len(first)))
    return first[:cut] + second[cut:], second[:cut] + first[cut:]


def mutate(mask, probability, rng):
    flips = rng.random(len(mask)) < probability
    mutated = tuple(int(bit != flip) for bit, flip in zip(mask, flips))
    return repair(mutated, rng)


def tournament(population, ranks, distances, rng):
    first, second = rng.choice(len(population), size=2, replace=False)
    first_key = (ranks[first], -distances[first])
    second_key = (ranks[second], -distances[second])
    if first_key == second_key:
        return population[first if rng.integers(2) == 0 else second]
    return population[first if first_key < second_key else second]


def rank_population(objectives):
    fronts = nondominated_sort(objectives)
    ranks = [0] * len(objectives)
    distances = {}
    for rank, front in enumerate(fronts):
        for index in front:
            ranks[index] = rank
        distances.update(crowding_distance(objectives, front))
    return fronts, ranks, distances


def survivors(population, objectives, size):
    selected = []
    for front in nondominated_sort(objectives):
        slots = size - len(selected)
        if len(front) > slots:
            distance = crowding_distance(objectives, front)
            selected.extend(sorted(front, key=distance.get, reverse=True)[:slots])
        else:
            selected.extend(front)
        if len(selected) == size:
            break
    return [population[index] for index in selected]


def pareto_points(objectives):
    """Extract distinct points in increasing feature count for two objectives."""
    front = []
    best_error = float("inf")
    for errors, count in sorted(set(objectives), key=lambda point: (point[1], point[0])):
        if errors < best_error:
            front.append((errors, count))
            best_error = errors
    return front


def hypervolume(front, validation_size, n_features):
    # Normalized area in the box with reference (validation_size, n_features + 1).
    previous_error = validation_size
    area = 0
    for errors, count in front:
        area += (previous_error - errors) * (n_features + 1 - count)
        previous_error = errors
    return area / (validation_size * n_features)


def search(fitness, n_features, validation_size, seed, config):
    size = config["population_size"]
    generations = config["generations"]
    crossover_probability = config["crossover_probability"]
    mutation_probability = config["mutation_probability"]
    if (size < 2 or generations < 1 or not 0 <= crossover_probability <= 1
            or not 0 <= mutation_probability <= 1):
        raise ValueError("Invalid NSGA-II parameters.")
    rng = np.random.default_rng(seed)
    population = initialize(size, n_features, rng)
    scores = {}
    history = []

    def evaluate(masks):
        for mask in masks:
            if mask not in scores:
                scores[mask] = fitness(mask)
        return [scores[mask] for mask in masks]

    objectives = evaluate(population)
    for generation in range(generations + 1):
        front = pareto_points(scores.values())
        history.append({
            "generation": generation,
            "evaluations": size * (generation + 1),
            "unique_masks": len(scores),
            "pareto_points": len(front),
            "best_validation_accuracy": 1 - min(point[0] for point in front) / validation_size,
            "min_features": front[0][1],
            "hypervolume": hypervolume(front, validation_size, n_features),
        })
        if generation == generations:
            break

        _, ranks, distances = rank_population(objectives)
        children = []
        while len(children) < size:
            first = tournament(population, ranks, distances, rng)
            second = tournament(population, ranks, distances, rng)
            offspring = crossover(first, second, crossover_probability, rng)
            for child in offspring[:size - len(children)]:
                children.append(mutate(child, mutation_probability, rng))

        combined = population + children
        population = survivors(combined, objectives + evaluate(children), size)
        objectives = [scores[mask] for mask in population]

    return SearchResult(population, scores, history)
