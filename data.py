from dataclasses import dataclass

import numpy as np


N_FEATURES = 18
# Columns define five linear combinations of the informative features.
REDUNDANT_WEIGHTS = (
    (1.0, 0.0, 0.0, 0.0, 0.5),
    (0.5, 1.0, 0.0, 0.0, 0.0),
    (0.0, -0.6, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.5, 1.0, 0.0),
    (0.0, 0.0, 0.0, -0.7, 1.0),
)


@dataclass(frozen=True)
class Dataset:
    features: np.ndarray
    labels: np.ndarray
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    feature_types: tuple[str, ...]


def generate_dataset(config):
    samples = config["samples"]
    train_fraction = config["train_fraction"]
    validation_fraction = config["validation_fraction"]
    signal = np.asarray(config["signal"], dtype=float)
    weights = np.asarray(REDUNDANT_WEIGHTS)
    noise_features = config["noise_features"]
    redundant_noise = config["redundant_noise"]
    if not (0 < train_fraction and 0 < validation_fraction
            and train_fraction + validation_fraction < 1):
        raise ValueError("Train, validation and test must have positive fractions.")
    if (samples < 6 or signal.shape != (weights.shape[0],)
            or not np.all(np.isfinite(signal)) or not np.all(signal != 0)
            or noise_features < 1 or signal.size + weights.shape[1] + noise_features != N_FEATURES
            or not np.isfinite(redundant_noise) or redundant_noise < 0):
        raise ValueError("Invalid parameters for the 18-feature dataset.")

    rng = np.random.default_rng(config["seed"])
    labels = rng.permutation(np.arange(samples) % 2)
    informative = rng.normal(size=(samples, signal.size))
    informative += (labels[:, None] - 0.5) * signal
    redundant = informative @ weights
    redundant += rng.normal(scale=redundant_noise, size=redundant.shape)
    noise = rng.normal(size=(samples, noise_features))
    features = np.column_stack((informative, redundant, noise))
    feature_types = (("informative",) * signal.size
                     + ("redundant",) * weights.shape[1] + ("noise",) * noise_features)

    split_rng = np.random.default_rng(config["split_seed"])
    splits = [[], [], []]
    for label in (0, 1):
        indices = split_rng.permutation(np.flatnonzero(labels == label))
        train_end = round(indices.size * train_fraction)
        validation_end = train_end + round(indices.size * validation_fraction)
        if not 0 < train_end < validation_end < indices.size:
            raise ValueError("Every split must contain examples of both classes.")
        for split, group in zip(splits, np.split(indices, [train_end, validation_end])):
            split.extend(group)
    train, validation, test = (split_rng.permutation(split) for split in splits)
    return Dataset(features, labels, train, validation, test, feature_types)


def error_count(train_x, train_y, evaluation_x, evaluation_y, mask, ridge):
    """Fit a regularized linear discriminant on the supplied training rows."""
    selected = np.asarray(mask, dtype=bool)
    train_x = train_x[:, selected]
    evaluation_x = evaluation_x[:, selected]
    center = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0] = 1
    train_x = (train_x - center) / scale
    evaluation_x = (evaluation_x - center) / scale

    negative, positive = train_x[train_y == 0], train_x[train_y == 1]
    mean_negative, mean_positive = negative.mean(axis=0), positive.mean(axis=0)
    residuals = np.vstack((negative - mean_negative, positive - mean_positive))
    covariance = (residuals.T @ residuals) / max(train_x.shape[0] - 2, 1)
    covariance += ridge * np.eye(selected.sum())
    direction = np.linalg.solve(covariance, mean_positive - mean_negative)
    midpoint = (mean_negative + mean_positive) / 2
    log_prior = np.log(len(positive) / len(negative))
    predicted = (evaluation_x - midpoint) @ direction + log_prior >= 0
    return int(np.count_nonzero(predicted != evaluation_y))


class ValidationFitness:
    def __init__(self, train_x, train_y, validation_x, validation_y, ridge):
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("The classifier ridge must be positive and finite.")
        self.train_x = train_x
        self.train_y = train_y
        self.validation_x = validation_x
        self.validation_y = validation_y
        self.ridge = ridge

    def __call__(self, mask):
        if len(mask) != N_FEATURES or not any(mask) or any(bit not in (0, 1) for bit in mask):
            raise ValueError("Fitness requires a nonempty binary mask of length 18.")
        errors = error_count(self.train_x, self.train_y, self.validation_x,
                             self.validation_y, mask, self.ridge)
        return errors, sum(mask)
