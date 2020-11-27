from kedro.pipeline import Pipeline, node

from .nodes import predict, report_metrics, train_model


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                train_model,
                ["train_df", "test_df", "parameters"],
                "mlmodel",
            ),
            node(
                predict,
                ["mlmodel", "dataset_total", "len_test","train_df","parameters"],
                "predicted_stock_price",
            ),
            node(report_metrics, ["predicted_stock_price", "train_df", "test_df", "parameters"], None),
        ]
    )
