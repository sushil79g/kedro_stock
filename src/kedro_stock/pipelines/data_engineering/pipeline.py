from kedro.pipeline import Pipeline, node

from .nodes import split_data


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                split_data,
                ["stock_train", "stock_test","params:test_data_ratio"],
                dict(
                   train_df = "train_df",
                   test_df = "test_df",
                   dataset_total = "dataset_total",
                   len_test= "len_test",
                ),
            )
        ]
    )
