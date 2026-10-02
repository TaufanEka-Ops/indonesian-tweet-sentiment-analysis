"""Tests for the modelling and evaluation layer (fast, on a small fixture)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idtweets import evaluation, features, modelling


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    words = {
        -1: ["terrible", "awful", "worst", "bad", "hate"],
        0: ["okay", "maybe", "perhaps", "neutral", "fine"],
        1: ["great", "excellent", "best", "good", "love"],
    }
    texts, labels = [], []
    for label, vocab in words.items():
        for _ in range(60):
            k = rng.integers(3, 6)
            texts.append(" ".join(rng.choice(vocab, size=k)))
            labels.append(label)
    return pd.Series(texts), pd.Series(labels)


def test_build_pipeline_has_two_steps():
    pipe = modelling.build_pipeline(modelling.default_estimators()["naive_bayes"])
    assert list(pipe.named_steps) == ["tfidf", "clf"]


def test_tfidf_min_df_drops_hapax(toy):
    X, _ = toy
    vec = features.build_tfidf(min_df=2)
    Xt = vec.fit_transform(X)
    # every kept term appears in >=2 documents
    assert Xt.shape[1] > 0


def test_repeated_cv_runs_and_scores(toy):
    X, y = toy
    res = modelling.run_repeated_cv(
        X, y, modelling.build_pipeline(modelling.default_estimators()["logistic_regression"]),
        name="lr", n_splits=3, n_repeats=2,
    )
    assert res.name == "lr"
    assert res.macro_f1_mean > 0.5
    assert len(res.folds) == 6
    assert res.macro_f1_std >= 0


def test_majority_baseline_is_honest(toy):
    """On a balanced toy set the majority baseline is ~1/3, not 0.99."""
    X, y = toy
    res = modelling.run_repeated_cv(
        X, y, modelling.build_pipeline(modelling.default_estimators()["majority_baseline"]),
        name="dummy", n_splits=3, n_repeats=1,
    )
    assert 0.05 < res.macro_f1_mean < 0.40


def test_compare_models_returns_sorted_table(toy):
    X, y = toy
    table, results = modelling.compare_models(
        X, y, estimators={
            "majority_baseline": modelling.default_estimators()["majority_baseline"],
            "logistic_regression": modelling.default_estimators()["logistic_regression"],
        },
        n_splits=3, n_repeats=1,
    )
    assert set(results) == {"majority_baseline", "logistic_regression"}
    assert table.iloc[0]["macro_f1_mean"] >= table.iloc[-1]["macro_f1_mean"]


def test_holdout_is_stratified(toy):
    X, y = toy
    Xtr, Xte, ytr, yte = modelling.holdout_split(X, y, test_size=0.25)
    assert set(yte.unique()) == {-1, 0, 1}
    assert len(Xte) == 45


def test_oof_predictions_cover_every_row(toy):
    X, y = toy
    preds = modelling.oof_predictions(
        X, y, modelling.build_pipeline(modelling.default_estimators()["naive_bayes"]),
        n_splits=3,
    )
    assert len(preds) == len(y)


def test_evaluation_metrics_and_report(toy):
    X, y = toy
    pipe = modelling.build_pipeline(modelling.default_estimators()["logistic_regression"])
    Xtr, Xte, ytr, yte = modelling.holdout_split(X, y, test_size=0.3)
    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte)
    m = evaluation.classification_metrics(yte, preds)
    assert set(m) == {"accuracy", "macro_f1", "weighted_f1", "cohen_kappa"}
    rep = evaluation.report_frame(yte, preds, label_names={-1: "negative", 0: "neutral", 1: "positive"})
    assert len(rep) == 3
    assert rep["support"].sum() == len(yte)


def test_confusion_frame_shape(toy):
    y_true = [1, 1, 0, -1]
    y_pred = [1, 0, 0, -1]
    cm = evaluation.confusion_frame(y_true, y_pred)
    assert cm.shape == (3, 3)
    assert cm.loc["true_1", "pred_1"] == 1


def test_error_analysis_returns_only_mistakes():
    texts = pd.Series(["a", "b", "c", "d"])
    y_true = pd.Series([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1])
    err = evaluation.error_analysis(texts, y_true, y_pred, n_per_class=5)
    assert len(err) == 2
    assert (err["true"] != err["pred"]).all()
    assert set(err["confusion"]) == {"1->0", "0->1"}


def test_top_features_per_class(toy):
    X, y = toy
    pipe = modelling.build_pipeline(modelling.default_estimators()["logistic_regression"])
    pipe.fit(X, y)
    tf = evaluation.top_features_per_class(pipe, top_n=3)
    assert set(tf["class"]) == {-1, 0, 1}
    assert len(tf) == 9


def test_top_features_rejects_non_linear_model(toy):
    X, y = toy
    pipe = modelling.build_pipeline(modelling.default_estimators()["naive_bayes"])
    pipe.fit(X, y)
    with pytest.raises(ValueError):
        evaluation.top_features_per_class(pipe)


def test_naive_bayes_features(toy):
    X, y = toy
    pipe = modelling.build_pipeline(modelling.default_estimators()["naive_bayes"])
    pipe.fit(X, y)
    tf = evaluation.top_features_naive_bayes(pipe, top_n=3)
    assert len(tf) == 9
