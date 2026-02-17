import pandas as pd
from windermere_project.labels import LabelBuilder, LabelConfig


def test_label_builder_strict_threshold():
    df = pd.DataFrame({"chl_ugL": [19.9, 20.0, 20.1]})
    lb = LabelBuilder(LabelConfig(chl_value_col="chl_ugL", threshold_ugL=20.0, label_col="y", strictly_greater=True))
    out = lb.build_from_feature_matrix(df)
    assert out["y"].tolist() == [0, 0, 1]
