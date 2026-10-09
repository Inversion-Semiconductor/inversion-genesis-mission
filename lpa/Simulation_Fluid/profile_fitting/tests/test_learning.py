from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from fludat_fit.cli_common import FitWindow
from fludat_fit.dataset import NozzleDataset
from fludat_fit.evaluate_models import main
from fludat_fit.families import get_families
from fludat_fit.fitting import MultiStartLocalFit
from fludat_fit.learning import (
    ConstantModel,
    DirectObjective,
    IndirectObjective,
    NearestNeighborModel,
    ParameterModel,
    PolynomialModel,
    TrainingSample,
    TrainingSet,
    build_training_set,
    compare_models,
    cross_validate,
    evaluate_model,
    get_model_factories,
    get_objective,
    merge_evaluations,
    reference_parameter_space,
    refine_predictions,
)
from fludat_fit.learning.plotting import plot_model_comparison, plot_training_history
from fludat_fit.sampling import ConditionRanges

FAMILY = "conical:supergaussian"


@pytest.fixture(scope="module")
def training_set(tmp_path_factory) -> TrainingSet:
    from .conftest import write_gaussian_cube

    path = write_gaussian_cube(tmp_path_factory.mktemp("cube") / "cube.h5")
    dataset = NozzleDataset.load(path)
    ranges = ConditionRanges.from_dataset(dataset, angle_range=(-20.0, 20.0))
    return build_training_set(
        dataset,
        ranges.sample(12, "lhs", seed=0),
        get_families([FAMILY])[0],
        ranges=ranges,
        window=FitWindow(max_points=120),
        scheme=MultiStartLocalFit(n_starts=1),
    )


# ------------------------------------------------------------------- data
def test_reference_space_is_the_envelope(training_set):
    lineouts = [s.lineout for s in training_set.samples]
    family = training_set.family
    envelope = reference_parameter_space(family, lineouts)
    for lineout in lineouts:
        own = family.parameter_space(lineout.summary())
        assert np.all(envelope.lower <= own.lower + 1e-15)
        assert np.all(envelope.upper >= own.upper - 1e-15)
    assert envelope.names == training_set.space.names
    with pytest.raises(ValueError):
        reference_parameter_space(family, [])


def test_training_set_contents(training_set):
    assert len(training_set) == 12
    assert training_set.dimension == 4
    assert training_set.conditions.shape == (12, 3)
    features = training_set.features
    assert features.shape == (12, 3)
    assert np.all(features >= 0.0) and np.all(features <= 1.0)
    assert training_set.has_targets
    targets = training_set.targets
    assert targets.shape == (12, 4)
    assert np.all(targets >= 0.0) and np.all(targets <= 1.0)
    assert np.all(training_set.reference_nrmse < 1e-3)  # noise-free Gaussian
    objective = training_set.fit_objective(0)
    assert objective is training_set.fit_objective(0)
    assert objective(targets[0]) < 1e-6
    # Features normalise back to the conditions.
    np.testing.assert_allclose(
        training_set.ranges.denormalize(features), training_set.conditions
    )


def test_split_folds_subset(training_set):
    train, test = training_set.split(0.25, seed=1)
    assert len(train) == 9 and len(test) == 3
    assert train.space is training_set.space and train.ranges is training_set.ranges
    seen = []
    for fold_train, fold_test in training_set.k_folds(3, seed=2):
        assert len(fold_train) + len(fold_test) == 12
        seen.extend(s.conditions for s in fold_test.samples)
    assert len(seen) == 12 and len(set(seen)) == 12
    with pytest.raises(ValueError):
        training_set.split(1.5)
    with pytest.raises(ValueError):
        list(training_set.k_folds(1))
    with pytest.raises(ValueError):
        TrainingSet(training_set.family, training_set.space, training_set.ranges, [])


def test_save_and_load_round_trip(training_set, tmp_path):
    path = training_set.save(tmp_path / "nested" / "set.npz")
    loaded = TrainingSet.load(path)
    assert loaded.family.name == FAMILY and loaded.name == training_set.name
    assert loaded.space.names == training_set.space.names
    np.testing.assert_allclose(loaded.space.lower, training_set.space.lower)
    np.testing.assert_allclose(loaded.targets, training_set.targets)
    np.testing.assert_allclose(loaded.conditions, training_set.conditions)
    np.testing.assert_allclose(loaded.reference_nrmse, training_set.reference_nrmse)
    np.testing.assert_allclose(
        loaded.samples[3].lineout.density, training_set.samples[3].lineout.density
    )
    assert (
        loaded.samples[3].reference_evaluations
        == training_set.samples[3].reference_evaluations
    )
    with pytest.raises(ValueError, match="not"):
        TrainingSet.load(path, family=get_families(["asymmetric_sine"])[0])

    # Sets without targets survive too.
    bare = TrainingSet(
        training_set.family,
        training_set.space,
        training_set.ranges,
        [TrainingSample(s.lineout) for s in training_set.samples],
    )
    assert not bare.has_targets
    with pytest.raises(ValueError, match="no local-fit targets"):
        bare.targets
    reloaded = TrainingSet.load(bare.save(tmp_path / "bare.npz"))
    assert not reloaded.has_targets and np.all(np.isnan(reloaded.reference_nrmse))


def test_build_without_targets_and_with_workers(small_cube_path):
    dataset = NozzleDataset.load(small_cube_path)
    ranges = ConditionRanges.from_dataset(dataset)
    family = get_families([FAMILY])[0]
    points = ranges.sample(3, "grid")
    bare = build_training_set(dataset, points, family, ranges=ranges, fit_targets=False)
    assert len(bare) == 4 and not bare.has_targets
    parallel = build_training_set(
        dataset,
        points,
        family,
        ranges=ranges,
        scheme=MultiStartLocalFit(n_starts=1),
        window=FitWindow(max_points=60),
        workers=2,
    )
    assert parallel.has_targets and len(parallel) == 4


# -------------------------------------------------------------- objectives
def test_direct_objective_is_the_local_fit_loss(training_set):
    direct = DirectObjective()
    targets = training_set.targets
    losses = direct.per_sample(targets, training_set)
    assert losses.shape == (12,)
    for i in range(3):
        assert losses[i] == pytest.approx(training_set.fit_objective(i)(targets[i]))
    centre = np.full_like(targets, 0.5)
    assert direct.loss(centre, training_set) > direct.loss(targets, training_set)
    # Subsets by index and shape checks.
    partial = direct.per_sample(targets[[2, 5]], training_set, indices=[2, 5])
    np.testing.assert_allclose(partial, losses[[2, 5]])
    with pytest.raises(ValueError, match="shape"):
        direct.per_sample(targets[:3], training_set)
    with pytest.raises(ValueError):
        DirectObjective(eps=0.0)


def test_direct_gradient_matches_finite_differences(training_set):
    direct = DirectObjective(eps=1e-4)
    u = np.clip(training_set.targets[:2] + 0.05, 0.05, 0.95)
    gradient = direct.gradient(u, training_set, indices=[0, 1])
    assert gradient.shape == (2, 4)
    objective = training_set.fit_objective(0)
    h = 1e-4
    for k in range(4):
        step = np.zeros(4)
        step[k] = h
        expected = (objective(u[0] + step) - objective(u[0] - step)) / (2 * h)
        assert gradient[0, k] == pytest.approx(expected, rel=1e-6, abs=1e-12)
    # At the cube boundary the stencil is one-sided but finite.
    edge = np.zeros((1, 4))
    assert np.all(np.isfinite(direct.gradient(edge, training_set, indices=[0])))


def test_indirect_objective(training_set):
    indirect = IndirectObjective()
    targets = training_set.targets
    assert np.all(indirect.per_sample(targets, training_set) == 0.0)
    shifted = np.clip(targets + 0.1, 0, 1)
    losses = indirect.per_sample(shifted, training_set)
    assert np.all(losses > 0.0) and np.all(losses <= 0.01 + 1e-12)
    gradient = indirect.gradient(shifted, training_set)
    np.testing.assert_allclose(gradient, 2.0 * (shifted - targets) / 4)
    weighted = IndirectObjective(weights=[1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        weighted.per_sample(shifted, training_set),
        (shifted[:, 0] - targets[:, 0]) ** 2 / 4,
    )
    with pytest.raises(ValueError):
        IndirectObjective(weights=[-1.0, 1, 1, 1])
    with pytest.raises(ValueError):
        IndirectObjective(weights=[1.0, 1.0]).per_sample(targets, training_set)

    bare = TrainingSet(
        training_set.family,
        training_set.space,
        training_set.ranges,
        [TrainingSample(s.lineout) for s in training_set.samples],
    )
    with pytest.raises(ValueError, match="needs local-fit targets"):
        indirect.check(bare)
    assert get_objective("direct").name == "direct"
    with pytest.raises(KeyError):
        get_objective("mystery")


# ------------------------------------------------------------------ models
def test_constant_model_under_both_objectives(training_set):
    indirect = ConstantModel()
    indirect.fit(training_set, IndirectObjective())
    np.testing.assert_allclose(indirect.value, training_set.targets.mean(axis=0))
    assert indirect.predict(training_set.features).shape == (12, 4)

    direct = ConstantModel(maxiter=50)
    direct.fit(training_set, DirectObjective())
    loss_direct = DirectObjective().loss(
        direct.predict(training_set.features), training_set
    )
    loss_mean = DirectObjective().loss(
        indirect.predict(training_set.features), training_set
    )
    assert loss_direct <= loss_mean + 1e-12
    with pytest.raises(RuntimeError):
        ConstantModel().predict(training_set.features)


def test_nearest_neighbour_model(training_set):
    model = NearestNeighborModel(k=1)
    assert model.name == "knn1"
    model.fit(training_set, IndirectObjective())
    np.testing.assert_allclose(
        model.predict(training_set.features), training_set.targets
    )
    smooth = NearestNeighborModel(k=3)
    smooth.fit(training_set, DirectObjective())  # objective is irrelevant for a lookup
    predictions = smooth.predict(training_set.features[:2] + 0.01)
    assert predictions.shape == (2, 4) and np.all(
        (0 <= predictions) & (predictions <= 1)
    )
    bare = TrainingSet(
        training_set.family,
        training_set.space,
        training_set.ranges,
        [TrainingSample(s.lineout) for s in training_set.samples],
    )
    with pytest.raises(ValueError, match="needs local-fit targets"):
        NearestNeighborModel().fit(bare, DirectObjective())
    with pytest.raises(ValueError):
        NearestNeighborModel(k=0)


def test_polynomial_model_fits_and_improves(training_set):
    model = PolynomialModel(degree=1, maxiter=50)
    assert model.basis(training_set.features).shape == (12, 4)
    assert PolynomialModel(degree=2).basis(training_set.features).shape == (12, 10)
    model.fit(training_set, IndirectObjective())
    predictions = model.predict(training_set.features)
    assert predictions.shape == (12, 4)
    assert IndirectObjective().loss(
        predictions, training_set
    ) < IndirectObjective().loss(np.full((12, 4), 0.5), training_set)
    assert model.history and model.history[-1] <= model.history[0] + 1e-12

    direct = PolynomialModel(degree=1, maxiter=30)
    direct.fit(training_set, DirectObjective())
    assert direct.history[-1] <= direct.history[0] + 1e-12
    assert (
        DirectObjective().loss(direct.predict(training_set.features), training_set)
        < 0.05
    )

    bare = TrainingSet(
        training_set.family,
        training_set.space,
        training_set.ranges,
        [TrainingSample(s.lineout) for s in training_set.samples],
    )
    cold = PolynomialModel(degree=0, maxiter=10)
    cold.fit(bare, DirectObjective())  # direct training needs no targets
    assert cold.predict(bare.features).shape == (12, 4)
    with pytest.raises(ValueError):
        PolynomialModel(degree=-1)


def test_model_registry_and_protocol():
    factories = get_model_factories()
    assert set(factories) == {"constant", "knn3", "poly1", "poly2"}
    for factory in factories.values():
        assert isinstance(factory(), ParameterModel)
    assert list(get_model_factories(["poly1"])) == ["poly1"]
    with pytest.raises(KeyError):
        get_model_factories(["gpt"])


# -------------------------------------------------------------- evaluation
def test_evaluate_model_scores_against_the_local_fit(training_set):
    train, test = training_set.split(0.25, seed=0)
    evaluation = evaluate_model(
        NearestNeighborModel(1), train, test, IndirectObjective(), refine=True
    )
    assert evaluation.model_name == "knn1" and evaluation.objective_name == "indirect"
    assert evaluation.training_size == len(train) and evaluation.n_test == len(test)
    assert evaluation.predictions.shape == (len(test), 4)
    assert evaluation.nrmse.shape == (len(test),)
    assert evaluation.direct_loss.shape == (len(test),)
    assert evaluation.parameter_error.shape == (len(test), 4)
    assert evaluation.indirect_loss.shape == (len(test),)
    assert np.all(evaluation.nrmse >= 0.0)
    # The refined fit is at least as good as the raw prediction.
    assert evaluation.refinement is not None
    assert np.all(evaluation.refinement.nrmse <= evaluation.nrmse + 1e-9)
    assert np.all(evaluation.refinement.evaluations > 0)
    summary = evaluation.summary()
    assert (
        summary["nrmse"]["median"] >= 0.0
        and summary["refinement"]["evaluation_ratio"] is not None
    )
    assert set(summary["parameter_error"]) == set(training_set.space.names)
    rows = evaluation.rows()
    assert (
        len(rows) == len(test)
        and "pred_fwhm" in rows[0]
        and "refined_evaluations" in rows[0]
    )
    json.dumps(summary)


def test_unsupported_combination_is_reported_not_raised(training_set):
    bare = TrainingSet(
        training_set.family,
        training_set.space,
        training_set.ranges,
        [TrainingSample(s.lineout) for s in training_set.samples],
    )
    train, test = bare.split(0.25)
    evaluation = evaluate_model(NearestNeighborModel(), train, test, DirectObjective())
    assert evaluation.error and "targets" in evaluation.error
    assert np.all(evaluation.predictions == 0.5)
    assert evaluation.parameter_error is None and evaluation.indirect_loss is None
    assert evaluation.summary()["indirect_loss"] is None

    class BadShape:
        name = "bad"

        def fit(self, training_set, objective):
            pass

        def predict(self, features):
            return np.zeros((1, 1))

    with pytest.raises(ValueError, match="shape"):
        evaluate_model(BadShape(), train, test, DirectObjective())


def test_cross_validation_and_merge(training_set):
    folds = cross_validate(
        lambda: ConstantModel(), training_set, IndirectObjective(), k=3
    )
    assert len(folds) == 3
    merged = merge_evaluations(folds)
    assert merged.n_test == 12 and merged.predictions.shape == (12, 4)
    assert merged.test_set.has_targets
    with pytest.raises(ValueError):
        merge_evaluations([])
    other = evaluate_model(
        *(ConstantModel(), *training_set.split(0.5)), DirectObjective()
    )
    with pytest.raises(ValueError, match="one model and objective"):
        merge_evaluations([folds[0], other])


def test_custom_model_plugs_into_compare_models(training_set, tmp_path):
    """The framework only needs fit/predict; here an oracle returns the targets."""

    class Oracle:
        name = "oracle"

        def fit(self, training_set, objective):
            self._lookup = {s.conditions: s.target for s in training_set.samples}

        def predict(self, features):
            return np.array([self._targets[i] for i in range(features.shape[0])])

    class CheatingOracle(Oracle):
        """Peeks at the test set through the fixed reference space: perfect predictions."""

        def fit(self, training_set, objective):
            self.full = training_set

        def predict(self, features):
            # Look up by feature vector among the full set (test features are known).
            out = []
            for row in features:
                index = int(np.argmin(np.linalg.norm(FULL.features - row, axis=1)))
                out.append(FULL.targets[index])
            return np.array(out)

    global FULL
    FULL = training_set
    comparison = compare_models(
        {"oracle": CheatingOracle, "constant": ConstantModel},
        training_set,
        [DirectObjective(), IndirectObjective()],
        folds=None,
        test_fraction=0.25,
        refine=True,
    )
    assert len(comparison.evaluations) == 4
    ranked = comparison.ranked()
    assert ranked[0].model_name == "oracle"
    assert np.all(ranked[0].indirect_loss == 0.0)
    table = comparison.table()
    assert "local fit" in table and "oracle" in table
    payload = comparison.to_dict()
    assert payload["parameters"] == list(training_set.space.names)
    assert [e["model"] for e in payload["evaluations"]][0] == "oracle"
    rows = comparison.rows()
    assert len(rows) == 4 * 3
    json.dumps(payload)

    figure = plot_model_comparison(comparison)
    assert len(figure.axes) == 2
    history_figure = plot_training_history(ranked[0], PolynomialModel())
    assert history_figure.axes[0].get_xlabel() == "objective call"

    kfold = compare_models(
        {"constant": ConstantModel}, training_set, [IndirectObjective()], folds=3
    )
    assert kfold.protocol.startswith("3-fold") and kfold.evaluations[0].n_test == 12


def test_refine_predictions_improves_on_the_centre(training_set):
    test = training_set.subset([0, 1])
    centre = np.full((2, 4), 0.5)
    refinement = refine_predictions(centre, test)
    assert refinement.nrmse.shape == (2,)
    assert np.all(refinement.evaluations > 0)
    assert refinement.evaluation_ratio.shape == (2,)


# --------------------------------------------------------------------- CLI
def test_cli_builds_caches_and_evaluates(small_cube_path, tmp_path, capsys):
    cache = tmp_path / "set.npz"
    json_path, csv_path, plot_path = (
        tmp_path / n for n in ("m.json", "m.csv", "m.png")
    )
    code = main(
        [
            str(small_cube_path),
            "--family",
            FAMILY,
            "--angle-range",
            "-10",
            "10",
            "--samples",
            "8",
            "--starts",
            "1",
            "--max-points",
            "80",
            "--training-set",
            str(cache),
            "--models",
            "constant",
            "knn3",
            "--folds",
            "0",
            "--test-fraction",
            "0.25",
            "--refine",
            "--quiet",
            "--json",
            str(json_path),
            "--csv",
            str(csv_path),
            "--plot",
            str(plot_path),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "local fit" in out and "knn3" in out
    assert cache.is_file() and json_path.is_file() and plot_path.is_file()
    payload = json.loads(json_path.read_text())
    assert payload["family"] == FAMILY and payload["n_samples"] == 8
    assert {e["model"] for e in payload["evaluations"]} == {"constant", "knn3"}
    assert {e["objective"] for e in payload["evaluations"]} == {"direct", "indirect"}
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4 * 2  # 4 evaluations x 2 test samples
    assert "refined_evaluations" in rows[0]

    # Second run: the cached set is loaded, no cube needed, direct only.
    code = main(
        [
            "--training-set",
            str(cache),
            "--models",
            "poly1",
            "--objectives",
            "direct",
            "--folds",
            "2",
            "--quiet",
        ]
    )
    assert code == 0
    assert "poly1" in capsys.readouterr().out


def test_cli_list_families_and_errors(capsys, small_cube_path, tmp_path):
    assert main(["--list-families"]) == 0
    assert FAMILY in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--samples", "4"])  # neither cube nor cached set
    with pytest.raises(SystemExit):
        main(
            [
                str(small_cube_path),
                "--models",
                "unknown",
                "--samples",
                "3",
                "--starts",
                "1",
                "--quiet",
            ]
        )
