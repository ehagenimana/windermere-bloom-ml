import pandas as pd
import numpy as np

from windermere_project.baselines import PersistenceBaseline, SeasonalMeanBaseline


def test_persistence_baseline_predicts_previous():
    df = pd.DataFrame(
        {
            "samplingPoint.notation": ["A", "A", "A"],
            "phenomenonTime": ["2005-01-01", "2005-01-02", "2005-01-03"],
            "y": [0, 1, 1],
        }
    )
    b = PersistenceBaseline().fit(df, y_col="y", site_col="samplingPoint.notation", time_col="phenomenonTime")
    r = b.predict(df, y_col="y", site_col="samplingPoint.notation", time_col="phenomenonTime")
    assert len(r.y_pred) == 3
    # second pred should equal first true
    assert r.y_pred[1] == 0


def test_seasonal_mean_baseline_train_only():
    train = pd.DataFrame({"month": [1, 1, 2, 2], "y": [0, 1, 0, 0]})
    test = pd.DataFrame({"month": [1, 2, 3], "y": [0, 0, 1]})

    b = SeasonalMeanBaseline(month_col="month").fit(train, y_col="y")
    r = b.predict(test, y_col="y")

    # month 1: p=0.5, month 2: p=0.0, month 3 uses global p = 0.25
    assert np.isclose(r.y_score[0], 0.5)
    assert np.isclose(r.y_score[1], 0.0)
    assert np.isclose(r.y_score[2], 0.25)
