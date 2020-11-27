from typing import Any, Dict
from io import StringIO
import pandas as pd

def build_csv(data):
    byte_to_string = StringIO(str(data.content, "utf-8"))
    df =  pd.read_csv(byte_to_string)
    return df

def split_data(traindata, testdata, split_ratio):
    train_df = build_csv(traindata)
    test_df = build_csv(testdata)
    training_set = train_df.iloc[:, 1:2].values
    testing_set = test_df.iloc[:, 1:2].values
    dataset_total =  pd.concat((train_df['Open'], test_df['Open']), axis=0)

    return dict(train_df = training_set,
                test_df = testing_set,
                dataset_total = dataset_total,
                len_test = len(test_df)
    )
