import inspect

import legacy_forest


class TestModelHyperparameters:
    """
    Guards for two settings that were measured, not guessed. Both were
    changed because the forest scored WORSE than a same-weekday-and-hour
    average (ratio 0.91); see the notes in legacy_forest.py.
    """

    def test_depth_is_capped(self):
        # Unconstrained, the forest memorised individual days: training error
        # ran at less than half the holdout error.
        assert legacy_forest.RF_PARAMS.get("max_depth") is not None
        assert legacy_forest.RF_PARAMS["max_depth"] <= 6

    def test_fit_and_evaluation_share_hyperparameters(self):
        """Otherwise the reported metrics describe a model nobody ships."""
        for fn in (legacy_forest._fit, legacy_forest._evaluate):
            source = inspect.getsource(fn)
            assert "RandomForestRegressor(**RF_PARAMS)" in source, fn.__name__

    def test_day_mean_temperature_is_not_a_feature(self):
        """
        Stored as a daily mean it was a unique fingerprint per day (61 days,
        61 distinct values, smallest gap 0.046 °C) rather than a weather
        signal. It may return once per-hour readings accumulate.
        """
        assert "temperature" not in legacy_forest.FEATURE_COLUMNS
