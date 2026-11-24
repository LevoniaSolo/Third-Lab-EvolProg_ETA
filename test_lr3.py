import json
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import numpy as np

from data import N_FEATURES, ValidationFitness, error_count, generate_dataset
from experiments import select_masks, summarize_runs, weight_interval
from nsga2 import (crowding_distance, crossover, dominates, hypervolume, initialize,
                   mutate, nondominated_sort, pareto_points, search, survivors, tournament)


CONFIG = json.loads(Path(__file__).with_name("config.json").read_text(encoding="utf-8"))
SMALL_SEARCH = {**CONFIG["search"], "population_size": 15, "generations": 6}


def make_fitness(data):
    return ValidationFitness(data.features[data.train], data.labels[data.train],
                             data.features[data.validation], data.labels[data.validation],
                             CONFIG["classifier"]["ridge"])


class NSGA2Tests(unittest.TestCase):
    def test_dominance_requires_a_strict_improvement(self):
        self.assertTrue(dominates((2, 3), (2, 4)))
        self.assertTrue(dominates((1, 3), (2, 4)))
        self.assertFalse(dominates((2, 4), (2, 4)))
        self.assertFalse(dominates((2, 4), (1, 5)))
        self.assertFalse(dominates((2, 4), (2, 3)))

    def test_nondominated_sort_with_duplicate_objectives(self):
        points = [(2, 4), (3, 2), (2, 4), (3, 4), (4, 5), (1, 6)]
        self.assertEqual(nondominated_sort(points), [[0, 1, 2, 5], [3], [4]])
        self.assertEqual(nondominated_sort([]), [])
        self.assertEqual(pareto_points(points), [(3, 2), (2, 4), (1, 6)])

    def test_crowding_distance_normalizes_each_axis(self):
        points = [(0, 10), (1, 7), (3, 4), (6, 3)]
        distances = crowding_distance(points, list(range(4)))
        self.assertEqual(distances[0], float("inf"))
        self.assertEqual(distances[3], float("inf"))
        self.assertAlmostEqual(distances[1], 3 / 6 + 6 / 7)
        self.assertAlmostEqual(distances[2], 5 / 6 + 4 / 7)

    def test_crowding_handles_constant_objectives_and_singletons(self):
        self.assertEqual(crowding_distance([(2, 3)] * 3, [0, 1, 2]), dict.fromkeys(range(3), 0))
        self.assertEqual(crowding_distance([(2, 3)], [0]), {0: float("inf")})

    def test_initialization_binary_masks_and_repair(self):
        rng = np.random.default_rng(5)
        population = initialize(100, N_FEATURES, rng)
        self.assertEqual(len(population), 100)
        self.assertTrue(all(len(mask) == N_FEATURES and any(mask) and set(mask) <= {0, 1}
                            for mask in population))
        self.assertEqual(initialize(8, 1, rng), [(1,)] * 8)
        self.assertEqual(sum(mutate((1,) * N_FEATURES, 1, rng)), 1)

    def test_crossover_and_mutation_respect_probabilities(self):
        rng = np.random.default_rng(9)
        first, second = (1, 1, 0, 0), (0, 0, 1, 1)
        self.assertEqual(crossover(first, second, 0, rng), (first, second))
        children = crossover(first, second, 1, rng)
        valid_pairs = [(first[:i] + second[i:], second[:i] + first[i:])
                       for i in range(1, len(first))]
        self.assertIn(children, valid_pairs)
        self.assertEqual(mutate(first, 0, rng), first)
        self.assertEqual(mutate(first, 1, rng), second)

    def test_tournament_prefers_rank_then_crowding(self):
        rng = np.random.default_rng(8)
        population = [(1, 0), (0, 1)]
        self.assertEqual(tournament(population, [0, 1], {0: 0, 1: float("inf")}, rng),
                         population[0])
        self.assertEqual(tournament(population, [0, 0], {0: 0, 1: 2}, rng), population[1])

    def test_elitist_survival_keeps_front_boundaries(self):
        population = [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 1)]
        objectives = [(7, 1), (4, 2), (2, 3), (9, 1)]
        self.assertEqual(survivors(population, objectives, 2), [population[0], population[2]])
        self.assertEqual(survivors(population, objectives, 3), population[:3])

    def test_archive_hypervolume(self):
        self.assertAlmostEqual(hypervolume([(3, 2), (1, 4)], 5, 4),
                               (2 * 3 + 2 * 1) / 20)


class SummaryTests(unittest.TestCase):
    def test_summary_describes_final_metrics_across_runs(self):
        runs = [
            {"hypervolume": 0.625, "best_validation_accuracy": 0.875, "min_features": 2},
            {"hypervolume": 0.875, "best_validation_accuracy": 0.75, "min_features": 4},
        ]
        expected = {
            "hypervolume": (0.625, 0.75, 0.75, 0.125, 0.875),
            "best_validation_accuracy": (0.75, 0.8125, 0.8125, 0.0625, 0.875),
            "min_features": (2, 3, 3, 1, 4),
        }
        rows = summarize_runs(runs)
        self.assertEqual({row["metric"] for row in rows}, set(expected))
        for row in rows:
            self.assertEqual(row["runs"], 2)
            self.assertEqual(row["ddof"], 0)
            values = tuple(row[key] for key in ("min", "mean", "median", "std", "max"))
            self.assertEqual(values, expected[row["metric"]])


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.data_config = {**CONFIG["data"], "samples": 120}
        self.data = generate_dataset(self.data_config)
        self.fitness = make_fitness(self.data)

    def run_search(self, fitness, seed=17):
        return search(fitness, N_FEATURES, len(self.data.validation), seed, SMALL_SEARCH)

    def test_dataset_is_reproducible_stratified_and_disjoint(self):
        again = generate_dataset(self.data_config)
        np.testing.assert_array_equal(self.data.features, again.features)
        np.testing.assert_array_equal(self.data.labels, again.labels)
        indices = [self.data.train, self.data.validation, self.data.test]
        for split, name, size in zip(indices, ("train", "validation", "test"), (72, 24, 24)):
            self.assertEqual(len(split), size)
            self.assertEqual(self.data.labels[split].sum(), size // 2)
            np.testing.assert_array_equal(split, getattr(again, name))
        np.testing.assert_array_equal(np.sort(np.concatenate(indices)), np.arange(120))
        self.assertEqual(self.data.features.shape, (120, 18))
        self.assertEqual(self.data.feature_types.count("informative"), 5)
        self.assertEqual(self.data.feature_types.count("redundant"), 5)
        self.assertEqual(self.data.feature_types.count("noise"), 8)

    def test_search_reproducibility_and_budget(self):
        result = self.run_search(self.fitness)
        self.assertEqual(result, self.run_search(make_fitness(self.data)))
        self.assertNotEqual(list(result.scores), list(self.run_search(self.fitness, seed=18).scores))
        self.assertEqual(len(result.population), 15)
        self.assertEqual(result.history[-1]["evaluations"], 15 * 7)
        self.assertEqual(result.history[-1]["unique_masks"], len(result.scores))
        volumes = [row["hypervolume"] for row in result.history]
        self.assertEqual(volumes, sorted(volumes))

    def test_fitness_uses_train_to_fit_and_validation_to_score(self):
        mask = (1,) + (0,) * (N_FEATURES - 1)
        with patch("data.error_count", wraps=error_count) as scorer:
            self.fitness(mask)
        train_x, train_y, evaluation_x, evaluation_y, _, _ = scorer.call_args.args
        np.testing.assert_array_equal(train_x, self.data.features[self.data.train])
        np.testing.assert_array_equal(train_y, self.data.labels[self.data.train])
        np.testing.assert_array_equal(evaluation_x, self.data.features[self.data.validation])
        np.testing.assert_array_equal(evaluation_y, self.data.labels[self.data.validation])
        with self.assertRaises(ValueError):
            self.fitness((0,) * N_FEATURES)

    def test_changing_test_data_does_not_change_search(self):
        changed_x, changed_y = self.data.features.copy(), self.data.labels.copy()
        changed_x[self.data.test] = np.nan
        changed_y[self.data.test] = 1 - changed_y[self.data.test]
        changed = replace(self.data, features=changed_x, labels=changed_y)
        self.assertEqual(self.run_search(self.fitness), self.run_search(make_fitness(changed)))

    def test_weighted_sum_distinguishes_unsupported_and_tied_points(self):
        front = [(10, 1), (9, 2), (4, 3)]
        self.assertIsNone(weight_interval(front[1], front, 20))
        threshold = Fraction(20, 71)
        self.assertEqual(weight_interval(front[0], front, 20), (0, threshold))
        self.assertEqual(weight_interval(front[-1], front, 20), (threshold, 1))
        collinear = [(10, 1), (7, 2), (4, 3)]
        self.assertEqual(weight_interval(collinear[1], collinear, 20), (threshold, threshold))

    def test_representative_masks_use_three_distinct_front_points(self):
        masks = [(1,) * count + (0,) * (N_FEATURES - count) for count in (1, 2, 3)]
        points = [(10, 1), (9, 2), (4, 3)]
        self.assertEqual(select_masks(dict(zip(masks, points)), points),
                         dict(zip(("min_features", "compromise", "max_quality"), masks)))


if __name__ == "__main__":
    unittest.main()
