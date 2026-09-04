import numpy as np

from salmon_price_estimator.models.xgboost_baseline import fit_xgboost, predict_one_step

PARAMS = {
    "max_depth": 3,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
}


def _synthetic_data(n: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.05, size=n)
    return X, y


def test_fit_xgboost_returns_booster_that_predicts():
    X, y = _synthetic_data()

    booster = fit_xgboost(X, y, PARAMS, num_boost_round=50)
    prediction = predict_one_step(booster, X[0])

    assert isinstance(prediction, float)
    assert np.isfinite(prediction)


def test_fit_xgboost_learns_the_relationship():
    X, y = _synthetic_data(n=300)

    booster = fit_xgboost(X, y, PARAMS, num_boost_round=100)
    predictions = np.array([predict_one_step(booster, row) for row in X])

    # Should fit the training data reasonably well - not a strict bound,
    # just a sanity check that the booster actually learned something.
    assert np.corrcoef(predictions, y)[0, 1] > 0.9
